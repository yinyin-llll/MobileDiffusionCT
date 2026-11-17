"""Process-local CUDA peaks and synchronized per-case inference timing."""
import csv
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

EFFICIENCY_FIELDS = [
    'TrainPeakAllocatedGiB', 'TrainPeakReservedGiB', 'TrainObservedProcessPeakGiB',
    'TrainWallHours', 'TestPeakAllocatedGiB', 'TestPeakReservedGiB',
    'TestObservedProcessPeakGiB', 'TestWallSec', 'E2ESecPerCase',
    'InferenceSec', 'InferenceSec_std', 'InferenceSec_ci95', 'InferenceSteadySec',
    'InferenceCases', 'TrainResourceSource', 'TestResourceSource', 'TimingContext']


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Need finite nonnegative durations')
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n > 1 else None
    return dict(n=n,mean=float(values.mean()),std=sd,ci95=half)


class RunPerformance:
    def __init__(self, phase, gpu_ids=(), started_at=None):
        self.phase = phase
        self.gpu_ids = list(gpu_ids)
        self.started = time.perf_counter() if started_at is None else started_at
        self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.cases = []
        for device in self.gpu_ids:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def synchronize(self):
        for device in self.gpu_ids:
            torch.cuda.synchronize(device)

    def begin_inference(self):
        self.synchronize()
        return time.perf_counter()

    def end_inference(self, started, case):
        self.synchronize()
        elapsed = time.perf_counter() - started
        self.cases.append(dict(case=case,inference_sec=elapsed))
        return elapsed

    def snapshot(self, status='running', **extra):
        self.synchronize()
        devices = [dict(device=device,name=torch.cuda.get_device_name(device),
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
                   for device in self.gpu_ids]
        result = dict(phase=self.phase,status=status,started_utc=self.started_utc,
                      wall_sec=time.perf_counter()-self.started,devices=devices,
                      memory_scope='CUDA allocator peaks since before model initialization; training includes validation/checkpoint phases',
                      timing_context='shared GPU / concurrent experiment suite; not an isolated speed benchmark',
                      inference_scope='synchronized model.test call; excludes data loading, host-to-device input transfer and external quality metrics; includes model-side postprocessing',
                      **extra)
        if self.cases:
            result['inference'] = timing_summary([r['inference_sec'] for r in self.cases])
            if len(self.cases)>1:
                result['inference_steady'] = timing_summary([r['inference_sec'] for r in self.cases[1:]])
            result['first_case_sec'] = self.cases[0]['inference_sec']
        return result

    def write(self, folder, status='running', **extra):
        folder = Path(folder)
        payload = self.snapshot(status,**extra)
        atomic_json(folder/'performance.json',payload)
        if self.cases:
            with (folder/'performance_per_case.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=['case','inference_sec'])
                writer.writeheader();writer.writerows(self.cases)
        if self.phase=='test':append_efficiency_to_wide_csv(folder)
        return payload


def efficiency_from_native(result_dir):
    result_dir=Path(result_dir)
    values={name:'' for name in EFFICIENCY_FIELDS}
    fallback=result_dir/'efficiency.json'
    if fallback.exists():
        values.update({k:v for k,v in json.loads(fallback.read_text()).items() if k in values})
    manifest=result_dir/'run_manifest.json'
    options=json.loads(manifest.read_text()).get('options',{}) if manifest.exists() else {}
    folders={'MultiViewCTDiffusion':'multiView_CTDiffusion','MultiViewCTGAN':'multiView_CTGAN',
             'X2CTOfficial':'official_X2CT_GAN','DPGANOfficial':'official_DP_GAN_B'}
    model=folders.get(options.get('model_class'))
    train_root=(Path(options.get('MODEL_SAVE_PATH','save_models'))/model/options.get('data','')/options.get('tag','')) if model else None
    for phase,path in [('Train',train_root/'performance.json' if train_root else None),('Test',result_dir/'performance.json')]:
        if path is None or not path.is_file():continue
        data=json.loads(path.read_text())
        devices=data.get('devices',[])
        if devices:
            values[phase+'PeakAllocatedGiB']=max(d['peak_allocated_gib'] for d in devices)
            values[phase+'PeakReservedGiB']=max(d['peak_reserved_gib'] for d in devices)
        if data.get('status')=='complete':
            values['TrainWallHours' if phase=='Train' else 'TestWallSec']=data['wall_sec']/3600 if phase=='Train' else data['wall_sec']
        values[phase+'ResourceSource']='native synchronized instrumentation'
        values['TimingContext']=data.get('timing_context','')
        if phase=='Test' and 'inference' in data:
            summary=data['inference'];values.update(InferenceSec=summary['mean'],InferenceSec_std=summary['std'],
                InferenceSec_ci95=summary['ci95'],InferenceCases=summary['n'],
                InferenceSteadySec=data.get('inference_steady',{}).get('mean',''))
        if phase=='Test' and data.get('status')=='complete' and data.get('evaluated_cases'):
            values['E2ESecPerCase']=data['wall_sec']/data['evaluated_cases']
    return values


def append_efficiency_to_wide_csv(result_dir):
    path=Path(result_dir)/'metrics_summary_wide.csv'
    if not path.exists():return
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)!=1:raise ValueError('Expected one wide summary row: '+str(path))
    row=rows[0];row.update(efficiency_from_native(result_dir))
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    temporary.replace(path)
