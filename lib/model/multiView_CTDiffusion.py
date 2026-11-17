# ------------------------------------------------------------------------------
# Conditional latent diffusion model for multi-view X-ray -> 3D CT generation.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn.functional as F
from lib.utils.projection import project_volume, uniform_angles
import lib.utils.metrics as Metrics
from lib.model.base_model import Base_Model
from lib.model.nets.utils import init_net
from lib.model.nets.generator.latent_diffusion_3d import MultiViewLatentDiffusion


class CTDiffusion(Base_Model):
  def __init__(self):
    super(CTDiffusion, self).__init__()
    self.use_ema = True
    self.ema_decay = 0.999
    self.ema_state = None

  @property
  def name(self):
    return 'multiView_CTDiffusion'

  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    memory_fraction = getattr(opt, 'diffusion_cuda_memory_fraction', 0.0)
    if memory_fraction:
      if not 0 < memory_fraction <= 1:
        raise ValueError('CUDA allocator fraction must be in (0, 1]')
      if self.gpu_ids:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, self.gpu_ids[0])

    self.loss_names = ['diffusion', 'recon', 'ae', 'structure', 'projection', 'intensity', 'detail', 'total']
    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    self.visual_names = ['G_real', 'G_fake', 'G_input1', 'G_input2', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']
    self.model_names = ['G']

    latent_channels = getattr(opt, 'diffusion_latent_channels', 64)
    cond_channels = getattr(opt, 'diffusion_cond_channels', 64)
    timesteps = getattr(opt, 'diffusion_timesteps', 1000)
    beta_start = getattr(opt, 'diffusion_beta_start', 1e-4)
    beta_end = getattr(opt, 'diffusion_beta_end', 2e-2)
    noise_schedule = getattr(opt, 'diffusion_noise_schedule', 'cosine')
    cond_drop_prob = getattr(opt, 'diffusion_cond_drop_prob', 0.2)

    cond_encoder_type = getattr(opt, 'diffusion_cond_encoder_type', 'unet')
    vit_depth = getattr(opt, 'diffusion_vit_depth', 1)
    vit_heads = getattr(opt, 'diffusion_vit_heads', 4)
    vit_mlp_ratio = getattr(opt, 'diffusion_vit_mlp_ratio', 4.0)
    vit_dropout = getattr(opt, 'diffusion_vit_dropout', 0.0)
    view_fusion = getattr(opt, 'diffusion_view_fusion', 'mean')
    view_pos_encoding = getattr(opt, 'diffusion_view_pos_encoding', 'none')
    backproject_angle_span = getattr(opt, 'diffusion_backproject_angle_span', 360.0)
    backproject_angle_offset = getattr(opt, 'diffusion_backproject_angle_offset', 0.0)
    aux_loss_mode = getattr(opt, 'diffusion_aux_loss_mode', 'all')
    aux_t_max_ratio = getattr(opt, 'diffusion_aux_t_max_ratio', 1.0)
    aux_snr_gamma = getattr(opt, 'diffusion_aux_snr_gamma', 5.0)

    net = MultiViewLatentDiffusion(
      latent_channels=latent_channels,
      cond_channels=cond_channels,
      timesteps=timesteps,
      beta_start=beta_start,
      beta_end=beta_end,
      noise_schedule=noise_schedule,
      cond_drop_prob=cond_drop_prob,
      cond_encoder_type=cond_encoder_type,
      vit_depth=vit_depth,
      vit_heads=vit_heads,
      vit_mlp_ratio=vit_mlp_ratio,
      vit_dropout=vit_dropout,
      view_fusion=view_fusion,
      view_pos_encoding=view_pos_encoding,
      backproject_angle_span=backproject_angle_span,
      backproject_angle_offset=backproject_angle_offset,
      aux_loss_mode=aux_loss_mode,
      aux_t_max_ratio=aux_t_max_ratio,
      aux_snr_gamma=aux_snr_gamma,
      separable_cond=getattr(opt, 'diffusion_separable_cond', False),
      separable_denoiser=getattr(opt, 'diffusion_separable_denoiser', False),
      condition_lifting=getattr(opt, 'diffusion_condition_lifting', 'legacy'),
      condition_detail=getattr(opt, 'diffusion_condition_detail', 'none'),
      convolution_scope=getattr(opt, 'diffusion_convolution_scope', 'legacy'),
      denoiser_type=getattr(opt, 'diffusion_denoiser_type', 'residual'))
    print('Convolution architecture:', net.architecture_spec)
    self.netG = init_net(net, opt.init_type, self.gpu_ids)
    if getattr(opt, 'diffusion_freeze_autoencoder', False):
      path = getattr(opt, 'diffusion_autoencoder_checkpoint', '')
      if self.training and not path:
        raise ValueError('Frozen AE training requires diffusion_autoencoder_checkpoint')
      if path:
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        ae_state = {k[len('autoencoder.'):]: v for k, v in state.items() if k.startswith('autoencoder.')}
        self._get_inner_net().autoencoder.load_state_dict(ae_state, strict=True)
      self._get_inner_net().autoencoder.requires_grad_(False)

    self.use_ema = getattr(opt, 'diffusion_use_ema', True)
    self.ema_decay = getattr(opt, 'diffusion_ema_decay', 0.999)
    if self.use_ema:
      self._init_ema_state()

  def init_loss(self, opt):
    Base_Model.init_loss(self, opt)
    self.loss_recon_lambda = getattr(opt, 'diffusion_recon_lambda', 1.0)
    self.loss_ae_lambda = getattr(opt, 'diffusion_ae_lambda', 0.5)
    self.loss_diffusion_lambda = getattr(opt, 'diffusion_noise_lambda', 1.0)
    self.loss_structure_lambda = getattr(opt, 'diffusion_structure_lambda', 0.0)
    self.loss_projection_lambda = getattr(opt, 'diffusion_projection_lambda', 0.0)
    if self.loss_projection_lambda and (not getattr(opt, 'diffusion_require_geometry', False) or getattr(opt, 'diffusion_projection_protocol', 'disabled') != 'parallel_mean_v2'):
      raise ValueError(
        'Projection loss is positive, but this configuration does not declare validated '
        'parallel_mean_v2 geometry. For revision experiments, use an '
        'experiment/revision_v2 YAML with data/revision_v2. For a legacy run, set '
        'diffusion_projection_lambda: 0.0; that run is not a projection-loss ablation.')
    self.loss_intensity_lambda = getattr(opt, 'diffusion_intensity_lambda', 0.0)
    self.loss_detail_lambda = getattr(opt, 'diffusion_detail_lambda', 0.0)
    if all(getattr(self, 'loss_' + name + '_lambda') == 0 for name in ('recon', 'ae', 'structure', 'projection', 'intensity', 'detail')) and not getattr(opt, 'diffusion_freeze_autoencoder', False):
      raise ValueError('Noise-only training needs a pretrained frozen autoencoder')
    weight_decay = getattr(opt, 'diffusion_weight_decay', 1e-4)
    self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=weight_decay)
    self.optimizers = [self.optimizer_G]

  def get_normalization_list(self):
    return [
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
      [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]]
    ]

  def set_input(self, input):
    xr = input[1]
    if isinstance(xr, (list, tuple)):
      self.G_input1 = xr[0].to(self.device)
      self.G_input2 = xr[1].to(self.device)
      self.G_views = None
    else:
      # xr: [B,V,1,H,W]
      self.G_views = xr.to(self.device)
      self.G_input1 = self.G_views[:, 0]
      self.G_input2 = self.G_views[:, 1] if self.G_views.shape[1] > 1 else self.G_views[:, 0]
    self.G_real = input[0].to(self.device)
    self.image_paths = input[2:3]
    geometry=input[3].to(self.device) if len(input)>3 else None
    self._get_inner_net().cond_encoder.input_geometry=geometry

  def output_map(self, v, dim):
    ori_dim = v.dim()
    if ori_dim == 4:
      map_v = torch.mean(torch.abs(v), dim=dim)
      return map_v.unsqueeze(1)
    if ori_dim == 5:
      map_v = torch.mean(torch.abs(v), dim=dim)
      return map_v
    raise NotImplementedError()

  def transition(self, predict):
    p_max, p_min = predict.max(), predict.min()
    if (p_max - p_min).abs() < 1e-6:
      return predict * 0
    return (predict - p_min) / (p_max - p_min)

  def ct_unGaussian(self, value):
    return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

  def xray_unGaussian(self, value, mean_std):
    return value * mean_std[1] + mean_std[0]

  def _ct_to_xray_projection(self, ct_norm, dim):
    # Work in normalized [0,1] intensity domain to avoid HU sign artifacts in projection supervision.
    ct_norm01 = torch.clamp(self.ct_unGaussian(ct_norm), 0.0, 1.0)
    return torch.mean(ct_norm01, dim=dim).unsqueeze(1)

  def _projection_consistency_loss(self):
    if self.loss_projection_lambda == 0:
      return self.G_fake.new_zeros(self.G_fake.shape[0])
    if getattr(self.opt, 'diffusion_projection_protocol', 'disabled') != 'parallel_mean_v2' or self.G_views is None:
      raise ValueError('Projection loss requires calibrated revision v2 DRRs; disable it for real X-rays/CT slices')
    angles = uniform_angles(self.G_views.shape[1], self.opt.diffusion_backproject_angle_span,
                            self.opt.diffusion_backproject_angle_offset)
    pred = project_volume(self.ct_unGaussian(self.G_fake).clamp(0, 1), angles)
    target = self.xray_unGaussian(self.G_views[:, :, 0], self.opt.XRAY1_MEAN_STD)
    return (pred - target).abs().flatten(1).mean(1)

  def _intensity_alignment_loss(self):
    pred = torch.clamp(self.ct_unGaussian(self.G_fake), 0.0, 1.0)
    tgt = torch.clamp(self.ct_unGaussian(self.G_real), 0.0, 1.0)

    pred_mean = pred.mean(dim=(1, 2, 3))
    tgt_mean = tgt.mean(dim=(1, 2, 3))
    pred_std = pred.std(dim=(1, 2, 3), unbiased=False)
    tgt_std = tgt.std(dim=(1, 2, 3), unbiased=False)

    loss_mean = torch.abs(pred_mean - tgt_mean)
    loss_std = torch.abs(pred_std - tgt_std)
    return loss_mean + 0.5 * loss_std

  def projection_visual(self):
    self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
    self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
    self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
    self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

  def post_process(self, attributes_name):
    # Clamp normalized tensor to the value range that maps to [0, 1] after unnormalization.
    ct_mean, ct_std = self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]
    if ct_std == 0:
      return
    lower = (0.0 - ct_mean) / ct_std
    upper = (1.0 - ct_mean) / ct_std
    for name in attributes_name:
      setattr(self, name, torch.clamp(getattr(self, name), lower, upper))

  def metrics_evaluation(self):
    g_fake_unNorm = self.ct_unGaussian(self.G_fake)
    g_real_unNorm = self.ct_unGaussian(self.G_real)
    self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
    self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

  def _get_inner_net(self):
    if isinstance(self.netG, torch.nn.DataParallel):
      return self.netG.module
    return self.netG

  def _get_latent_shape(self):
    d = int(self.G_real.shape[-3] // 8)
    h = int(self.G_real.shape[-2] // 8)
    w = int(self.G_real.shape[-1] // 8)
    return max(1, d), max(1, h), max(1, w)

  def _laplacian_3d(self, x):
    # x: [B, D, H, W]
    k = x.new_zeros((1, 1, 3, 3, 3))
    k[0, 0, 1, 1, 1] = -6.0
    k[0, 0, 0, 1, 1] = 1.0
    k[0, 0, 2, 1, 1] = 1.0
    k[0, 0, 1, 0, 1] = 1.0
    k[0, 0, 1, 2, 1] = 1.0
    k[0, 0, 1, 1, 0] = 1.0
    k[0, 0, 1, 1, 2] = 1.0
    return F.conv3d(x.unsqueeze(1), k, padding=1).squeeze(1)

  def _detail_loss_3d(self, pred, target):
    # Detail supervision in normalized [0,1] domain to focus on texture/high-frequency consistency.
    pred_n = torch.clamp(self.ct_unGaussian(pred), 0.0, 1.0)
    target_n = torch.clamp(self.ct_unGaussian(target), 0.0, 1.0)
    return torch.abs(self._laplacian_3d(pred_n) - self._laplacian_3d(target_n)).flatten(1).mean(1)

  def _gradient_loss_3d(self, pred, target):
    return sum((pred.diff(dim=dim)-target.diff(dim=dim)).abs().flatten(1).mean(1)
               for dim in (1, 2, 3)) / 3.0

  def _init_ema_state(self):
    inner_net = self._get_inner_net()
    self.ema_state = {k: v.detach().clone() for k, v in inner_net.state_dict().items()}

  @torch.no_grad()
  def _update_ema_state(self):
    if (not self.use_ema) or (self.ema_state is None):
      return
    inner_net = self._get_inner_net()
    for k, v in inner_net.state_dict().items():
      if k not in self.ema_state:
        self.ema_state[k] = v.detach().clone()
        continue
      if v.dtype.is_floating_point:
        self.ema_state[k].mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)
      else:
        self.ema_state[k].copy_(v.detach())

  def _swap_in_ema(self):
    inner_net = self._get_inner_net()
    self._backup_state = {k: v.detach().clone() for k, v in inner_net.state_dict().items()}
    inner_net.load_state_dict(self.ema_state, strict=False)

  def _restore_from_backup(self):
    if not hasattr(self, '_backup_state') or self._backup_state is None:
      return
    inner_net = self._get_inner_net()
    inner_net.load_state_dict(self._backup_state, strict=False)
    self._backup_state = None

  def forward(self):
    if self.training:
      if getattr(self, 'G_views', None) is not None:
        losses, self.G_fake = self.netG(self.G_real, self.G_views, None)
      else:
        losses, self.G_fake = self.netG(self.G_real, self.G_input1, self.G_input2)
      self.loss_diffusion = losses['diffusion'] * self.loss_diffusion_lambda
      aux_weight = losses.get('aux_weight', torch.tensor(1.0, device=self.device))
      self.loss_recon = (losses['recon'] * aux_weight).mean() * self.loss_recon_lambda
      self.loss_ae = (losses['ae'] * aux_weight).mean() * self.loss_ae_lambda
      self.loss_structure = (self._gradient_loss_3d(self.G_fake, self.G_real) * aux_weight).mean() * self.loss_structure_lambda
      self.loss_projection = (self._projection_consistency_loss() * aux_weight).mean() * self.loss_projection_lambda
      self.loss_intensity = (self._intensity_alignment_loss() * aux_weight).mean() * self.loss_intensity_lambda
      self.loss_detail = (self._detail_loss_3d(self.G_fake, self.G_real) * aux_weight).mean() * self.loss_detail_lambda
      self.loss_total = self.loss_diffusion + self.loss_recon + self.loss_ae + self.loss_structure + self.loss_projection + self.loss_intensity + self.loss_detail
    else:
      inner_net = self._get_inner_net()
      use_shadow = self.use_ema and (self.ema_state is not None)
      if use_shadow:
        self._swap_in_ema()
      try:
        sample_steps = getattr(self.opt, 'diffusion_sample_steps', 30)
        ddim_eta = getattr(self.opt, 'diffusion_ddim_eta', 0.0)
        cfg_scale = getattr(self.opt, 'diffusion_cfg_scale', 2.0)
        force_uncond = getattr(self.opt, 'diffusion_sample_uncond', False)
        latent_shape = self._get_latent_shape()
        if getattr(self, 'G_views', None) is not None:
          self.G_fake = inner_net.ddim_sample(self.G_views, None, latent_shape, sample_steps=sample_steps, eta=ddim_eta, cfg_scale=cfg_scale, force_uncond=force_uncond)
        else:
          self.G_fake = inner_net.ddim_sample(self.G_input1, self.G_input2, latent_shape, sample_steps=sample_steps, eta=ddim_eta, cfg_scale=cfg_scale, force_uncond=force_uncond)
      finally:
        if use_shadow:
          self._restore_from_backup()

      self.loss_diffusion = torch.tensor(0.0, device=self.device)
      self.loss_recon = torch.tensor(0.0, device=self.device)
      self.loss_ae = torch.tensor(0.0, device=self.device)
      self.loss_structure = torch.tensor(0.0, device=self.device)
      self.loss_projection = torch.tensor(0.0, device=self.device)
      self.loss_intensity = torch.tensor(0.0, device=self.device)
      self.loss_detail = torch.tensor(0.0, device=self.device)
      self.loss_total = torch.tensor(0.0, device=self.device)
      self.post_process(['G_fake'])
      self.projection_visual()
      self.metrics_evaluation()

  def optimize_parameters(self):
    self()
    self.optimizer_G.zero_grad()
    self.loss_total.backward()
    self.optimizer_G.step()
    self._update_ema_state()

  def save_networks(self, which_epoch, total_steps, latest=False):
    for name in self.model_names:
      if not isinstance(name, str):
        continue
      if latest:
        save_filename = '{}_net_{}.pth'.format('latest', name)
        save_path = os.path.join(self.save_dir, save_filename)
      else:
        save_filename = '{}_net_{}.pth'.format(which_epoch, name)
        save_path = os.path.join(self.save_dir, str(which_epoch))
        if not os.path.exists(save_path):
          os.makedirs(save_path)
        save_path = os.path.join(save_path, save_filename)

      net = getattr(self, 'net' + name)
      if len(self.gpu_ids) > 0 and torch.cuda.is_available():
        param_dict = net.module.cpu().state_dict()
        net.cuda(self.device)
      else:
        param_dict = net.cpu().state_dict()

      ema_cpu = None
      if self.use_ema and self.ema_state is not None:
        ema_cpu = {k: v.detach().cpu() for k, v in self.ema_state.items()}

      lr = self.optimizers[0].param_groups[0]['lr'] if len(self.optimizers) > 0 else 0.0
      save_dict = {
        'iters': total_steps,
        'epoch': which_epoch,
        'state_dict': param_dict,
        'ema_state': ema_cpu,
        'lr': lr,
      }
      torch.save(save_dict, save_path)

  def load_networks(self, which_epoch, load_Path=None, latest=False):
    total_steps = 0
    epoch_count = 0
    for name in self.model_names:
      if not isinstance(name, str):
        continue
      if latest:
        load_filename = '{}_net_{}.pth'.format('latest', name)
        load_path = os.path.join(load_Path, load_filename) if load_Path is not None else os.path.join(self.save_dir, load_filename)
      else:
        load_filename = '{}_net_{}.pth'.format(which_epoch, name)
        load_path = os.path.join(load_Path, str(which_epoch), load_filename) if load_Path is not None else os.path.join(self.save_dir, str(which_epoch), load_filename)

      net = getattr(self, 'net' + name)
      if isinstance(net, torch.nn.DataParallel):
        net = net.module

      print('**loading the model from %s' % load_path)
      save_dict = torch.load(load_path, map_location=self.device)
      state_dict = save_dict['state_dict']

      for key in list(state_dict.keys()):
        self._Base_Model__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

      will_state_dict = {i: j for i, j in state_dict.items() if i in net.state_dict().keys()}
      print('**loading {} parameters from {}(saved model), net size = {}'.format(len(will_state_dict), len(state_dict), len(net.state_dict())))
      net.load_state_dict(will_state_dict)
      net.to(self.device)

      if self.use_ema:
        ckpt_ema = save_dict.get('ema_state', None)
        if ckpt_ema is not None:
          self.ema_state = {k: v.to(self.device) for k, v in ckpt_ema.items()}
        else:
          self._init_ema_state()

      total_steps = save_dict.get('iters', 0)
      epoch_count = save_dict.get('epoch', 0)

    return total_steps, epoch_count
