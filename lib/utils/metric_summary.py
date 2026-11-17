"""Per-case test metric summaries with sample SD and Student-t confidence intervals."""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

METRICS = [
    ('MAE0','MAE0'),('MSE0','MSE0'),('MAE','MAE'),('MSE','MSE'),
    ('CosineSimilarity','CosineSimilarity'),('psnr-3d','PSNR-3d'),
    ('PSNR-1','PSNR-1'),('PSNR-2','PSNR-2'),('PSNR-3','PSNR-3'),('PSNR-avg','PSNR_avg'),
    ('SSIM-1','SSIM-1'),('SSIM-2','SSIM-2'),('SSIM-3','SSIM-3'),('SSIM-avg','SSIM-avg'),('NRMSE','NRMSE')]


def summarize_records(records):
    if not records:raise ValueError('No case metrics')
    if len({r['case'] for r in records})!=len(records):raise ValueError('Duplicate case IDs')
    rows=[]
    optional=[]
    if any('LPIPS' in r for r in records):
        if not all('LPIPS' in r for r in records):raise ValueError('LPIPS missing for part of the cohort')
        optional.append(('LPIPS','LPIPS'))
    for key,label in METRICS+optional:
        values=np.asarray([float(r[key]) if key in r else np.sqrt(float(r['MSE0'])) if key=='NRMSE' else float(r[key]) for r in records],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite metric: '+key)
        n=len(values);mean=float(values.mean());std=float(values.std(ddof=1)) if n>1 else 0.
        ci=float(stats.t.ppf(.975,n-1)*std/np.sqrt(n)) if n>1 else 0.
        rows.append(dict(metric=label,source_metric=key,n=n,mean=mean,std=std,ci95=ci,
            ci95_low=mean-ci,ci95_high=mean+ci))
    return rows


def wide_row(summary):
    return {field:row[value] for row in summary for field,value in
        ((row['metric'],'mean'),(row['metric']+'_std','std'),(row['metric']+'_ci95','ci95'))}


def write_summary(records,out_dir,print_table=True):
    summary=summarize_records(records);out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/'metrics_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    wide=wide_row(summary)
    from lib.utils.performance import efficiency_from_native
    wide.update(efficiency_from_native(out_dir))
    with (out_dir/'metrics_summary_wide.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(wide));writer.writeheader();writer.writerow(wide)
    if print_table:
        print('### All test metrics (case-level sample std; ci95 is Student-t 95% half-width)')
        print('| metric | mean | std | ci95 |')
        print('| --- | ---: | ---: | ---: |')
        for row in summary:print('| {} | {:.7g} | {:.7g} | {:.7g} |'.format(row['metric'],row['mean'],row['std'],row['ci95']))
    return summary
