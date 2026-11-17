# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils.visualizer import tensor_back_to_unnormalization, tensor_back_to_unMinMax
from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity, \
  Peak_Signal_to_Noise_Rate_3D
import copy
import tqdm
import torch
import numpy as np
import os
import csv
import time
from scipy import stats
from pathlib import Path
from lib.utils.revision_protocol import seed_all, seed_case, write_manifest
from lib.utils.performance import RunPerformance


def parse_args():
  parse = argparse.ArgumentParser(description='MobileDiffusionCT')
  parse.add_argument('--data', type=str, default='LIDC256', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='mobilect_standard_aplat2v', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='./data/DRR_APLAT_2V', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='test', dest='dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='splits/DRR_APLAT_2V/test.txt', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default='configs/standard_aplat_2v.yml', dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='align_ct_drr_multiview_std', dest='dataset_class',
                     help='Dataset class should select from unalign /')
  parse.add_argument('--model_class', type=str, default='MultiViewCTDiffusion', dest='model_class',
                     help='Model class should select from cyclegan / ')
  parse.add_argument('--check_point', type=str, default='1000', dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--how_many', type=int, dest='how_many', default=0,
                     help='0 evaluates the entire split; positive values are smoke tests only')
  parse.add_argument('--sample-steps', type=int, default=None)
  parse.add_argument('--cfg-scale', type=float, default=None)
  parse.add_argument('--unconditional', action='store_true')
  parse.add_argument('--condition-mode', choices=['original','swap','repeat_ap','repeat_lat'], default='original')
  parse.add_argument('--save-volumes', action='store_true')
  parse.add_argument('--lpips', action='store_true', help='Pretrained AlexNet LPIPS, all slices of all three planes')
  parse.add_argument('--lpips-device', default='cpu')
  parse.add_argument('--profile-flops', action='store_true', help='Profile one additional full reconstruction outside timing')
  parse.add_argument('--inference-warmup', action='store_true', help='One full warmup before measuring the first case')
  parse.add_argument('--resultdir', type=str, default='./results', dest='resultdir',
                     help='dir to save result')
  parse.add_argument('--seed', type=int, default=None, dest='seed',
                     help='optional random seed for DDIM initial noise and reproducible repeated sampling')
  args = parse.parse_args()
  return args


def evaluate(args):
  task_started = time.perf_counter()
  if args.seed is not None:
    seed_all(args.seed)

  # check gpu
  if args.gpuid == '':
    args.gpu_ids = []
  else:
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      raise RuntimeError('CUDA is unavailable; use --gpu "" for CPU evaluation')

  # check point
  if args.check_point is None:
    args.epoch_count = 1
  else:
    args.epoch_count = int(args.check_point)

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  if args.sample_steps is not None:
    opt.diffusion_sample_steps = args.sample_steps
  if args.cfg_scale is not None:
    opt.diffusion_cfg_scale = args.cfg_scale
  if args.unconditional:
    opt.diffusion_sample_uncond = True
  print_easy_dict(opt)

  opt.serial_batches = True

  # add data_augmentation
  datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = dataTestClass

  # get dataset
  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  if dataset_size == 0 or len(set(dataset.dataset_paths)) != len(dataset.dataset_paths):
    raise ValueError('Empty or duplicate evaluation case list')
  print('#Test images = %d' % dataset_size)

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))

  # set to test
  gan_model.eval()

  performance = RunPerformance('test', args.gpu_ids, started_at=task_started)
  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # must set to test Mode again, due to  omission of assigning mode to network layers
  # model.training is test, but BN.training is training
  if opt.verbose:
    print('## Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  if 'batch' in opt.norm_G:
    gan_model.eval()
  elif 'instance' in opt.norm_G:
    gan_model.eval()
    # instance norm in training mode is better
    for name, m in gan_model.named_modules():
      if m.__class__.__name__.startswith('InstanceNorm'):
        m.train()
  else:
    raise NotImplementedError()

  if opt.verbose:
    print('## Change to Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)
  #test_90 %s_%s
  result_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  if os.path.exists(os.path.join(result_dir, 'metrics_per_case.csv')):
    raise FileExistsError('Choose a new --resultdir; existing metrics will not be overwritten')
  write_manifest(os.path.join(result_dir, 'run_manifest.json'), opt, status='running')

  avg_dict = dict()
  case_records = []
  perceptual = None
  if args.lpips:
    from lib.utils.comparison_metrics import VolumeLPIPS
    perceptual = VolumeLPIPS(args.lpips_device)
  for epoch_i, data in tqdm.tqdm(enumerate(dataloader)):
    if args.how_many > 0 and epoch_i >= args.how_many:
      break
    if args.seed is not None:
      seed_all(seed_case(args.seed, Path(data[2][0]).parent.name))
    if args.condition_mode != 'original':
      ct, views, paths = data
      indices = {'swap': [1, 0], 'repeat_ap': [0, 0], 'repeat_lat': [1, 1]}[args.condition_mode]
      if isinstance(views, (list, tuple)):
        if len(views) != 2: raise ValueError('Condition diagnostics require two views')
        views = [views[k] for k in indices]
      else:
        if views.shape[1] != 2: raise ValueError('Condition diagnostics require two views')
        views = views[:, indices]
      data = (ct, views, paths)
    gan_model.set_input(data)
    if epoch_i == 0 and args.inference_warmup:
      gan_model.test()
      if args.seed is not None:
        seed_all(seed_case(args.seed, Path(data[2][0]).parent.name))
    inference_started = performance.begin_inference()
    gan_model.test()
    inference_sec = performance.end_inference(inference_started, Path(data[2][0]).parent.name)
    print('Inference {:.6f} sec/case (CUDA synchronized)'.format(inference_sec))

    visuals = gan_model.get_current_visuals()
    img_path = gan_model.get_image_paths()

    #
    # Evaluate Part
    #
    generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
    real_CT = visuals['G_real'].data.clone().cpu().numpy()
    # To [0, 1]
    # To NDHW
    if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
      generate_CT_transpose = generate_CT
      real_CT_transpose = real_CT
    else:
      generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
      real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
    generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                           opt.CT_MEAN_STD[1])
    real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
    # clip generate_CT
    generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)
    lpips_value = perceptual(real_CT_transpose[0], generate_CT_transpose[0]) if perceptual is not None else None
    if args.save_volumes:
      volume_dir = Path(result_dir) / 'volumes'
      volume_dir.mkdir(exist_ok=True)
      case_id = Path(img_path[0][0]).parent.name
      np.save(volume_dir / (case_id + '.npy'), generate_CT_transpose[0])

    # CT range 0-1
    mae0 = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse0 = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
    cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
    ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
    # CT range 0-4096
    generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.float64)
    real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.float64)
    psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=float(opt.CT_MIN_MAX[1] - opt.CT_MIN_MAX[0]))
    psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=float(opt.CT_MIN_MAX[1] - opt.CT_MIN_MAX[0]))
    mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)

    name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
    name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
    name = name2
    print(cosinesimilarity, name)
    if not np.all(np.isfinite(cosinesimilarity)):
      raise ValueError('Nonfinite cosine similarity: ' + name)

    metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MSE', mse), ('CosineSimilarity', cosinesimilarity),
                    ('psnr-3d', psnr_3d), ('PSNR-1', psnr[0]),
                    ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]),
                    ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3]), ('NRMSE', np.sqrt(mse0))]

    record = {'case': name}
    if lpips_value is not None:
      metrics_list.append(('LPIPS', [lpips_value]))
    for key, value in metrics_list:
      if not np.all(np.isfinite(value)):
        raise ValueError('Nonfinite metric {} for {}'.format(key, name))
      value_list = np.asarray(value).reshape(-1).tolist()
      if avg_dict.get(key) is None:
        avg_dict[key] = [] + value_list
      else:
        avg_dict[key].extend(value_list)
      record[key] = float(np.mean(value_list))
    case_records.append(record)

    if epoch_i == 0 and args.profile_flops:
      import json
      from lib.utils.comparison_metrics import profile_reconstruction
      flop_report = profile_reconstruction(gan_model.test)
      flop_report.update(case=name, model_class=opt.model_class,
                         Params=sum(p.numel() for p in gan_model.netG.parameters()),
                         sample_steps=getattr(opt, 'diffusion_sample_steps', None),
                         cfg_scale=getattr(opt, 'diffusion_cfg_scale', None))
      Path(result_dir, 'flops.json').write_text(json.dumps(flop_report, indent=2))
      print('### Full reconstruction FLOPs (T):', flop_report['FLOPs_T'])

    del visuals, img_path

  if len(case_records) > 0:
    csv_path = os.path.join(result_dir, 'metrics_per_case.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
      fieldnames = ['case'] + [k for k, _ in metrics_list]
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(case_records)
    print('### per-case metrics saved to {}'.format(csv_path))

  from lib.utils.metric_summary import write_summary
  summary = write_summary(case_records, result_dir)
  avg_dict = {row['source_metric']: row['mean'] for row in summary}

  resource_report = performance.write(result_dir, status='complete', evaluated_cases=len(case_records))
  print('### Runtime and memory: {}'.format(resource_report))

  write_manifest(os.path.join(result_dir, 'run_manifest.json'), opt,
                 status='complete', evaluated_cases=len(case_records),
                 performance_file='performance.json',
                 lpips_protocol='AlexNet-v0.1; all slices; equally averaged axial/coronal/sagittal; grayscale RGB; [0,1] to [-1,1]' if args.lpips else None,
                 flops_protocol='full reconstruction; MAC x2; convolution/GEMM/attention; separate pass' if args.profile_flops else None,
                 metric_protocol='slice_ssim_2d; PSNR_range=CT_MAX-CT_MIN; float_intensities')
  return avg_dict


if __name__ == '__main__':
  args = parse_args()
  evaluate(args)
