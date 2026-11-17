"""Train/save/load/test the actual entry points with synthetic HDF5 cases only."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import yaml


ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='verification/smoke_test_report.json');a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='mobilect_smoke_') as temporary:
        work=Path(temporary);data=work/'data';data.mkdir()
        cfg=yaml.safe_load((ROOT/'configs/standard_aplat_2v.yml').read_text())
        rng=np.random.default_rng(42)
        for name in ['synthetic_train','synthetic_val','synthetic_test']:
            folder=data/name;folder.mkdir()
            ct=rng.uniform(-900.,900.,size=(32,32,32)).astype(np.float32)
            projections=np.stack([ct.mean(1),ct.mean(2)])
            projections=(projections-projections.min())/max(float(np.ptp(projections)),1e-6)
            with h5py.File(folder/'ct_xray_data.h5','w') as f:
                f.create_dataset('ct',data=ct);f.create_dataset('xrays',data=projections)
                f.create_dataset('angles',data=np.array([0.,90.],np.float32))
                f.attrs['input_type']=cfg['DIFFUSION']['drr_input_type'];f.attrs['num_views']=2
        splits={}
        for phase in ['train','val','test']:
            path=work/(phase+'.txt');path.write_text('synthetic_'+phase+'\n');splits[phase]=str(path)
        cfg['MODEL_SAVE_PATH']=str(work/'save_models')
        cfg['TRAIN'].update(niter=1,niter_decay=0,nThreads=0,save_epoch_freq=1,print_freq=1,lr_policy='invariant')
        cfg.setdefault('TEST',{})['howmany_in_train']=0
        cfg['DATA_AUG'].update(fine_size=32,ct_channel=32,resize_size=32)
        cfg['DIFFUSION'].update(diffusion_timesteps=10,diffusion_sample_steps=2,diffusion_cuda_memory_fraction=0.)
        config=work/'smoke.yml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        common=['--ymlpath',str(config),'--dataroot',str(data),'--gpu','','--tag','synthetic_smoke',
                '--data','LIDC256','--dataset_class','align_ct_drr_multiview_std','--model_class','MultiViewCTDiffusion']
        train=[sys.executable,str(ROOT/'train.py'),*common,'--datasetfile',splits['train'],
               '--valid_datasetfile',splits['val'],'--valid_dataset','val','--seed','2026']
        test=[sys.executable,str(ROOT/'test.py'),*common,'--datasetfile',splits['test'],
              '--check_point','1','--seed','42','--profile-flops','--save-volumes','--resultdir',str(work/'results')]
        for name,command in [('train',train),('test',test)]:
            print('Running synthetic',name,'on CPU',flush=True)
            done=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if done.returncode:
                print(done.stdout[-12000:],flush=True);raise RuntimeError(name+' failed')
            print(done.stdout[-1800:],flush=True)
        checkpoint=work/'save_models/multiView_CTDiffusion/LIDC256/synthetic_smoke/checkpoint/1/1_net_G.pth'
        assert checkpoint.is_file()
        result=work/'results/LIDC256/test_1'
        with (result/'metrics_per_case.csv').open() as f:records=list(csv.DictReader(f))
        assert len(records)==1 and records[0]['case']=='synthetic_test'
        assert all(math.isfinite(float(value)) for key,value in records[0].items() if key!='case')
        manifest=json.loads((result/'run_manifest.json').read_text());assert manifest['status']=='complete'
        flops=json.loads((result/'flops.json').read_text());assert flops['Params']==17536467 and flops['FLOPs_T']>0
        volume=np.load(result/'volumes/synthetic_test.npy');assert volume.shape==(32,32,32) and np.isfinite(volume).all()
        model_root=checkpoint.parents[2];assert (model_root/'validation/validation_history.csv').is_file()
        assert list((model_root/'train_log').glob('events.out.tfevents.*'))
        report=dict(status='passed',device='cpu',synthetic_data_only=True,resolution=[32,32,32],
                    test_cases=1,training_epochs=1,parameters=flops['Params'],
                    checks=['HDF5 loading','forward/backward and optimizer','validation','TensorBoard logging',
                            'checkpoint save/load','sampling','all15metrics','FLOPs profiling','volume export'],
                    note='Reduced32^3/two-step smoke only; reference128^3/300-step configs are unchanged.')
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
