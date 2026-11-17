from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
  out = a.gather(0, t)
  return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def timestep_embedding(timesteps, dim):
  half = dim // 2
  exponent = -math.log(10000) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
  emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if dim % 2 == 1:
    emb = F.pad(emb, (0, 1), mode='constant', value=0)
  return emb


def view_angle_embedding(num_views, dim, device, dtype, angle_span=360.0, angle_offset=0.0):
  """Sinusoidal view-angle embedding for uniformly sampled DRR views."""
  half = dim // 2
  angles = torch.arange(num_views, device=device, dtype=dtype) / max(num_views, 1)
  angles = (angles * angle_span + angle_offset) * (math.pi / 180.0)
  freq = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1))
  emb = angles.unsqueeze(1) * freq.unsqueeze(0)
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if dim % 2 == 1:
    emb = F.pad(emb, (0, 1), mode='constant', value=0)
  return emb.view(1, num_views, dim, 1, 1)


def backproject_multiview_features(fv, target_shape, angle_span=360.0, angle_offset=0.0):
  """Approximate parallel-beam backprojection from 2D view features to a 3D latent grid.

  fv is [B,V,C,Hf,Wf]. The detector vertical coordinate is aligned with the
  CT superior-inferior/depth axis, while the detector horizontal coordinate is
  obtained by rotating the in-plane H/W coordinates according to each view angle.
  This is intended as a geometry-aware condition injection ablation, not as a
  calibrated cone-beam projector.
  """
  b, v, c, _, _ = fv.shape
  d, h, w = target_shape
  device, dtype = fv.device, fv.dtype

  z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
  y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
  x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
  zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

  vols = []
  for i in range(v):
    angle = float(angle_offset) + float(angle_span) * float(i) / max(v, 1)
    theta = math.radians(angle)
    u = xx * math.cos(theta) + yy * math.sin(theta)
    grid = torch.stack([u, zz], dim=-1).view(1, d * h * w, 1, 2).repeat(b, 1, 1, 1)
    sampled = F.grid_sample(fv[:, i], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    vols.append(sampled.view(b, c, d, h, w))
  return torch.stack(vols, dim=1).mean(dim=1)

def lift_observation_features(fv, target_shape, geometry):
  """Lift projections by angle and axial slices by position; geometry is [B,V,3].

  Channels: kind (0=parallel projection, 1=axial slice), angle in degrees,
  normalized slice depth. Slice features have a Gaussian plane support with
  sigma=one latent-grid interval; this is a feature prior, not CT inversion.
  """
  b,v,c,_,_=fv.shape
  if tuple(geometry.shape)!=(b,v,3):raise ValueError('Expected B,V,3 geometry')
  geometry=geometry.to(device=fv.device,dtype=fv.dtype)
  if not torch.isfinite(geometry).all():raise ValueError('Nonfinite geometry')
  d,h,w=target_shape
  z=torch.linspace(-1.,1.,d,device=fv.device,dtype=fv.dtype)
  y=torch.linspace(-1.,1.,h,device=fv.device,dtype=fv.dtype)
  x=torch.linspace(-1.,1.,w,device=fv.device,dtype=fv.dtype)
  zz,yy,xx=torch.meshgrid(z,y,x,indexing='ij')
  total=fv.new_zeros((b,c,d,h,w))
  for i in range(v):
    kind=geometry[:,i,0]
    if torch.all(kind==0):
      angle=geometry[:,i,1]*(math.pi/180.)
      u=xx[None]*angle.cos()[:,None,None,None]+yy[None]*angle.sin()[:,None,None,None]
      grid=torch.stack((u,zz[None].expand(b,-1,-1,-1)),dim=-1).reshape(b,d*h*w,1,2)
      sample=F.grid_sample(fv[:,i],grid,mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,c,d,h,w)
    elif torch.all(kind==1):
      position=geometry[:,i,2]
      if torch.any(position.abs()>1):raise ValueError('Slice outside normalized CT grid')
      plane=F.interpolate(fv[:,i],size=(h,w),mode='bilinear',align_corners=True)
      sigma=2./max(d-1,1)
      weight=torch.exp(-.5*((z[None]-position[:,None])/sigma)**2)
      sample=plane[:,:,None]*weight[:,None,:,None,None]
    else:raise ValueError('Unknown or inconsistent observation kind in batch')
    total=total+sample
  return total/max(v,1)


def cosine_beta_schedule(timesteps, s=0.008):
  steps = timesteps + 1
  x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
  alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clamp(betas, 1e-8, 0.999).float()


class DepthwiseSeparableConv(nn.Module):
  """Spatial depthwise filtering followed by linear 1x1 channel mixing."""
  def __init__(self, conv):
    super().__init__()
    if not isinstance(conv, (nn.Conv2d, nn.Conv3d)) or conv.groups != 1:
      raise ValueError('Expected a dense 2D/3D convolution')
    cls = type(conv)
    self.depthwise = cls(conv.in_channels, conv.in_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                         groups=conv.in_channels, bias=False, padding_mode=conv.padding_mode)
    self.pointwise = cls(conv.in_channels, conv.out_channels, kernel_size=1,
                         bias=conv.bias is not None)
    self.to(device=conv.weight.device, dtype=conv.weight.dtype)

  def forward(self, x):
    return self.pointwise(self.depthwise(x))


def replace_spatial_convolutions(module, dimensions):
  """Replace spatial dense convolutions; keep 1x1 projections and transposed convs."""
  cls = nn.Conv2d if dimensions == 2 else nn.Conv3d
  replaced = 0
  for name, child in list(module.named_children()):
    if isinstance(child, cls) and child.groups == 1 and any(k > 1 for k in child.kernel_size):
      setattr(module, name, DepthwiseSeparableConv(child))
      replaced += 1
    else:
      replaced += replace_spatial_convolutions(child, dimensions)
  return replaced


class ConvBlock2D(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1):
    super(ConvBlock2D, self).__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
      nn.GroupNorm(8, out_ch),
      nn.SiLU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
      nn.GroupNorm(8, out_ch),
      nn.SiLU(inplace=True),
    )

  def forward(self, x):
    return self.block(x)


class ConvBlock3D(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1):
    super(ConvBlock3D, self).__init__()
    self.block = nn.Sequential(
      nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
      nn.GroupNorm(8, out_ch),
      nn.SiLU(inplace=True),
      nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
      nn.GroupNorm(8, out_ch),
      nn.SiLU(inplace=True),
    )

  def forward(self, x):
    return self.block(x)


class ResidualBlock3D(nn.Module):
  def __init__(self, channels, time_dim):
    super(ResidualBlock3D, self).__init__()
    self.norm1 = nn.GroupNorm(8, channels)
    self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
    self.norm2 = nn.GroupNorm(8, channels)
    self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
    self.time_proj = nn.Linear(time_dim, channels)

  def forward(self, x, t_emb):
    h = self.conv1(F.silu(self.norm1(x)))
    h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    h = self.conv2(F.silu(self.norm2(h)))
    return x + h


class MultiViewConditionEncoder(nn.Module):
  def __init__(self, in_channels=1, base_channels=32, out_channels=64, view_fusion='mean', view_pos_encoding='none',
               backproject_angle_span=360.0, backproject_angle_offset=0.0):
    super(MultiViewConditionEncoder, self).__init__()
    self.view_fusion = view_fusion
    self.view_pos_encoding = view_pos_encoding
    self.backproject_angle_span = float(backproject_angle_span)
    self.backproject_angle_offset = float(backproject_angle_offset)
    self.view1 = nn.Sequential(
      ConvBlock2D(in_channels, base_channels, stride=2),
      ConvBlock2D(base_channels, base_channels * 2, stride=2),
      ConvBlock2D(base_channels * 2, out_channels, stride=1),
    )
    self.view2 = nn.Sequential(
      ConvBlock2D(in_channels, base_channels, stride=2),
      ConvBlock2D(base_channels, base_channels * 2, stride=2),
      ConvBlock2D(base_channels * 2, out_channels, stride=1),
    )
    self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
    if self.view_fusion == 'attention':
      self.view_cross_attn = nn.MultiheadAttention(out_channels, 1, batch_first=True)
      self.view_attn = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(out_channels, max(out_channels // 4, 8)),
        nn.SiLU(inplace=True),
        nn.Linear(max(out_channels // 4, 8), 1),
      )
    elif self.view_fusion not in ['mean', 'backprojection']:
      raise ValueError('Unsupported view fusion: {}'.format(self.view_fusion))

    if self.view_pos_encoding not in ['none', 'sin']:
      raise ValueError('Unsupported view position encoding: {}'.format(self.view_pos_encoding))

  def _fuse_multiview(self, fv):
    # fv: [B, V, C, H, W]. Mean is kept as the default for backward-compatible checkpoints.
    if self.view_pos_encoding == 'sin':
      fv = fv + view_angle_embedding(fv.shape[1], fv.shape[2], fv.device, fv.dtype, self.backproject_angle_span, self.backproject_angle_offset)
    if self.view_fusion == 'backprojection':
      raise RuntimeError('Backprojection fusion should call _backproject_multiview instead of _fuse_multiview.')
    if self.view_fusion == 'mean':
      return fv.mean(dim=1)
    b, v, c, h, w = fv.shape
    tokens = fv.mean(dim=(-2, -1))
    context, _ = self.view_cross_attn(tokens, tokens, tokens, need_weights=False)
    fv = fv + context[:, :, :, None, None]
    scores = self.view_attn(fv.reshape(b * v, c, h, w)).view(b, v, 1, 1, 1)
    weights = torch.softmax(scores, dim=1)
    return torch.sum(fv * weights, dim=1)

  def forward(self, xray1, xray2, target_shape):
    d, h, w = target_shape
    # multiview mode: xray1 is [B,V,1,H,W], xray2 is None
    if xray2 is None and xray1.dim() == 5:
      b, v, c, hh, ww = xray1.shape
      xv = xray1.view(b * v, c, hh, ww)
      ftmp = self.view1(xv)
      fv_raw = ftmp.view(b, v, ftmp.shape[1], ftmp.shape[-2], ftmp.shape[-1])
      if self.view_fusion == 'backprojection':
        geometry=getattr(self,'input_geometry',None)
        if geometry is not None:return lift_observation_features(fv_raw,target_shape,geometry)
        return backproject_multiview_features(fv_raw, target_shape, self.backproject_angle_span, self.backproject_angle_offset)
      fv = self._fuse_multiview(fv_raw)
      f = F.adaptive_avg_pool2d(fv, (h, w)).unsqueeze(2).repeat(1, 1, d, 1, 1)
      return f

    # legacy two-view mode
    f1 = self.view1(xray1)
    f2 = self.view2(xray2)
    if self.view_fusion == 'backprojection':
      fv_raw = torch.stack([f1, f2], dim=1)
      return backproject_multiview_features(fv_raw, target_shape, 180.0, 0.0)
    f1 = F.adaptive_avg_pool2d(f1, (h, w)).unsqueeze(2).repeat(1, 1, d, 1, 1)
    f2 = F.adaptive_avg_pool2d(f2, (h, w)).unsqueeze(2).repeat(1, 1, d, 1, 1)
    return self.fuse(torch.cat([f1, f2], dim=1))




class ViT2DConditionBlock(nn.Module):
  def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
    super(ViT2DConditionBlock, self).__init__()
    if dim % num_heads != 0:
      for h in [8, 4, 2, 1]:
        if dim % h == 0:
          num_heads = h
          break
    self.norm1 = nn.LayerNorm(dim)
    self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
    self.norm2 = nn.LayerNorm(dim)
    hidden = int(dim * mlp_ratio)
    self.mlp = nn.Sequential(
      nn.Linear(dim, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, dim),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    # x: [B, C, H, W]
    b, c, h, w = x.shape
    t = x.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
    t2 = self.norm1(t)
    a, _ = self.attn(t2, t2, t2, need_weights=False)
    t = t + a
    t = t + self.mlp(self.norm2(t))
    return t.transpose(1, 2).view(b, c, h, w)


class ViTConditionEncoder(nn.Module):
  def __init__(self, in_channels=1, base_channels=32, out_channels=64, depth=1, heads=4, mlp_ratio=4.0,
               dropout=0.0, view_fusion='mean', view_pos_encoding='none', backproject_angle_span=360.0,
               backproject_angle_offset=0.0):
    super(ViTConditionEncoder, self).__init__()
    self.view_fusion = view_fusion
    self.view_pos_encoding = view_pos_encoding
    self.backproject_angle_span = float(backproject_angle_span)
    self.backproject_angle_offset = float(backproject_angle_offset)
    self.stem1 = nn.Sequential(
      nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
      nn.GroupNorm(8, base_channels),
      nn.SiLU(inplace=True),
      nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, out_channels),
      nn.SiLU(inplace=True),
    )
    self.stem2 = nn.Sequential(
      nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
      nn.GroupNorm(8, base_channels),
      nn.SiLU(inplace=True),
      nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, out_channels),
      nn.SiLU(inplace=True),
    )
    self.vit1 = nn.Sequential(*[ViT2DConditionBlock(out_channels, num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(max(1, depth))])
    self.vit2 = nn.Sequential(*[ViT2DConditionBlock(out_channels, num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(max(1, depth))])
    self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
    if self.view_fusion == 'attention':
      self.view_cross_attn = nn.MultiheadAttention(out_channels, 1, batch_first=True)
      self.view_attn = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(out_channels, max(out_channels // 4, 8)),
        nn.SiLU(inplace=True),
        nn.Linear(max(out_channels // 4, 8), 1),
      )
    elif self.view_fusion not in ['mean', 'backprojection']:
      raise ValueError('Unsupported view fusion: {}'.format(self.view_fusion))

    if self.view_pos_encoding not in ['none', 'sin']:
      raise ValueError('Unsupported view position encoding: {}'.format(self.view_pos_encoding))

  def _fuse_multiview(self, fv):
    # fv: [B, V, C, H, W]
    if self.view_pos_encoding == 'sin':
      fv = fv + view_angle_embedding(fv.shape[1], fv.shape[2], fv.device, fv.dtype, self.backproject_angle_span, self.backproject_angle_offset)
    if self.view_fusion == 'backprojection':
      raise RuntimeError('Backprojection fusion should call backproject_multiview_features instead of _fuse_multiview.')
    if self.view_fusion == 'mean':
      return fv.mean(dim=1)
    b, v, c, h, w = fv.shape
    tokens = fv.mean(dim=(-2, -1))
    context, _ = self.view_cross_attn(tokens, tokens, tokens, need_weights=False)
    fv = fv + context[:, :, :, None, None]
    scores = self.view_attn(fv.reshape(b * v, c, h, w)).view(b, v, 1, 1, 1)
    weights = torch.softmax(scores, dim=1)
    return torch.sum(fv * weights, dim=1)

  def forward(self, xray1, xray2, target_shape):
    d, h, w = target_shape
    if xray2 is None and xray1.dim() == 5:
      b, v, c, hh, ww = xray1.shape
      xv = xray1.view(b * v, c, hh, ww)
      f = self.vit1(self.stem1(xv))
      fv_raw = f.view(b, v, f.shape[1], f.shape[-2], f.shape[-1])
      if self.view_fusion == 'backprojection':
        geometry=getattr(self,'input_geometry',None)
        if geometry is not None:return lift_observation_features(fv_raw,target_shape,geometry)
        return backproject_multiview_features(fv_raw, target_shape, self.backproject_angle_span, self.backproject_angle_offset)
      f = self._fuse_multiview(fv_raw)
      return F.adaptive_avg_pool2d(f, (h, w)).unsqueeze(2).repeat(1, 1, d, 1, 1)

    f1 = self.vit1(self.stem1(xray1))
    f2 = self.vit2(self.stem2(xray2))
    if self.view_fusion == 'backprojection':
      fv_raw = torch.stack([f1, f2], dim=1)
      return backproject_multiview_features(fv_raw, target_shape, 180.0, 0.0)
    f1 = F.adaptive_avg_pool2d(f1, (h, w)).unsqueeze(2).repeat(1, 1, d, 1, 1)
    f2 = F.adaptive_avg_pool2d(f2, (h, w)).unsqueeze(2).repeat(1, 1, d, 1, 1)
    return self.fuse(torch.cat([f1, f2], dim=1))


class CTLatentAutoEncoder3D(nn.Module):
  def __init__(self, in_channels=1, latent_channels=64):
    super(CTLatentAutoEncoder3D, self).__init__()
    self.enc1 = ConvBlock3D(in_channels, 32, stride=2)
    self.enc2 = ConvBlock3D(32, 64, stride=2)
    self.enc3 = ConvBlock3D(64, latent_channels, stride=2)
    self.bottleneck = ConvBlock3D(latent_channels, latent_channels)

    self.dec3 = nn.Sequential(
      nn.ConvTranspose3d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, 64),
      nn.SiLU(inplace=True),
    )
    self.dec2 = nn.Sequential(
      nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, 32),
      nn.SiLU(inplace=True),
    )
    self.dec1 = nn.Sequential(
      nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, 16),
      nn.SiLU(inplace=True),
      nn.Conv3d(16, 1, kernel_size=3, padding=1),
    )

  def encode(self, x):
    x = self.enc1(x)
    x = self.enc2(x)
    x = self.enc3(x)
    x = self.bottleneck(x)
    return x

  def decode(self, z):
    z = self.dec3(z)
    z = self.dec2(z)
    z = self.dec1(z)
    return z


class LatentDenoiser3D(nn.Module):
  def __init__(self, latent_channels=64, cond_channels=64, hidden_channels=128, time_dim=256):
    super(LatentDenoiser3D, self).__init__()
    self.time_dim = time_dim
    self.time_mlp = nn.Sequential(
      nn.Linear(time_dim, time_dim),
      nn.SiLU(inplace=True),
      nn.Linear(time_dim, time_dim),
    )
    self.in_proj = nn.Conv3d(latent_channels + cond_channels, hidden_channels, kernel_size=3, padding=1)
    self.res1 = ResidualBlock3D(hidden_channels, time_dim)
    self.res2 = ResidualBlock3D(hidden_channels, time_dim)
    self.res3 = ResidualBlock3D(hidden_channels, time_dim)
    self.out_norm = nn.GroupNorm(8, hidden_channels)
    self.out_proj = nn.Conv3d(hidden_channels, latent_channels, kernel_size=3, padding=1)

  def forward(self, z_t, cond, t):
    t_emb = timestep_embedding(t, self.time_dim)
    t_emb = self.time_mlp(t_emb)
    h = self.in_proj(torch.cat([z_t, cond], dim=1))
    h = self.res1(h, t_emb)
    h = self.res2(h, t_emb)
    h = self.res3(h, t_emb)
    h = self.out_proj(F.silu(self.out_norm(h)))
    return h


class MultiViewLatentDiffusion(nn.Module):
  def __init__(self, latent_channels=64, cond_channels=64, timesteps=1000, beta_start=1e-4, beta_end=2e-2,
               noise_schedule='linear', cond_drop_prob=0.0, cond_encoder_type='unet', vit_depth=1, vit_heads=4,
               vit_mlp_ratio=4.0, vit_dropout=0.0, view_fusion='mean', view_pos_encoding='none',
               backproject_angle_span=360.0, backproject_angle_offset=0.0, aux_loss_mode='all',
               aux_t_max_ratio=1.0, aux_snr_gamma=5.0, separable_cond=False, separable_denoiser=False,
               condition_lifting='legacy', condition_detail='none', convolution_scope='legacy',
               denoiser_type='residual'):
    super(MultiViewLatentDiffusion, self).__init__()
    self.latent_channels = latent_channels
    self.timesteps = timesteps
    self.noise_schedule = noise_schedule
    self.cond_drop_prob = float(cond_drop_prob)
    self.view_fusion = view_fusion
    self.view_pos_encoding = view_pos_encoding
    self.aux_loss_mode = str(aux_loss_mode)
    self.aux_t_max_ratio = float(aux_t_max_ratio)
    self.aux_snr_gamma = float(aux_snr_gamma)

    self.autoencoder = CTLatentAutoEncoder3D(in_channels=1, latent_channels=latent_channels)
    if cond_encoder_type == 'vit':
      self.cond_encoder = ViTConditionEncoder(in_channels=1, base_channels=32, out_channels=cond_channels, depth=vit_depth, heads=vit_heads, mlp_ratio=vit_mlp_ratio, dropout=vit_dropout, view_fusion=view_fusion, view_pos_encoding=view_pos_encoding, backproject_angle_span=backproject_angle_span, backproject_angle_offset=backproject_angle_offset)
    else:
      self.cond_encoder = MultiViewConditionEncoder(in_channels=1, base_channels=32, out_channels=cond_channels, view_fusion=view_fusion, view_pos_encoding=view_pos_encoding, backproject_angle_span=backproject_angle_span, backproject_angle_offset=backproject_angle_offset)
    if denoiser_type == 'residual':
      denoiser_class = LatentDenoiser3D
    elif denoiser_type == 'unet3d':
      from .latent_unet_3d import LatentUNetDenoiser3D
      denoiser_class = LatentUNetDenoiser3D
    else:
      raise ValueError('Unknown denoiser type: ' + str(denoiser_type))
    self.denoiser = denoiser_class(
      latent_channels=latent_channels,
      cond_channels=cond_channels,
      hidden_channels=max(latent_channels * 2, 64),
      time_dim=256)

    if convolution_scope not in ('legacy', 'all_separable', 'all_standard'):
      raise ValueError('Unknown convolution scope: ' + str(convolution_scope))
    legacy_conversion = convolution_scope == 'legacy'
    self.architecture_spec = dict(separable_cond=bool(separable_cond),
        separable_denoiser=bool(separable_denoiser),
        cond_replacements=replace_spatial_convolutions(self.cond_encoder, 2) if separable_cond and legacy_conversion else 0,
        denoiser_replacements=replace_spatial_convolutions(self.denoiser, 3) if separable_denoiser and legacy_conversion else 0)
    if denoiser_type != 'residual':
      self.architecture_spec['denoiser_type'] = denoiser_type

    if condition_lifting != 'legacy':
      if cond_encoder_type != 'vit' or view_fusion != 'backprojection':
        raise ValueError('Experimental lifting requires a ViT encoder and geometry fusion')
      from .residual_feature_lifting import ResidualPyramidConditionEncoder
      self.cond_encoder = ResidualPyramidConditionEncoder(self.cond_encoder, cond_channels, condition_lifting)
      self.architecture_spec['condition_lifting'] = condition_lifting

    self.condition_refiner = None
    if condition_detail != 'none':
      from .shape_texture_refiner import ShapeTextureConditionRefiner
      self.condition_refiner = ShapeTextureConditionRefiner(cond_channels, condition_detail)
      self.architecture_spec['condition_detail'] = condition_detail

    if not legacy_conversion:
      from .global_convolutions import apply_global_convolution_policy
      self.architecture_spec['convolution_scope'] = convolution_scope
      self.architecture_spec['whole_model_conversion'] = apply_global_convolution_policy(self, convolution_scope)

    if noise_schedule == 'cosine':
      betas = cosine_beta_schedule(timesteps)
    elif noise_schedule == 'linear':
      betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    else:
      raise ValueError('Unsupported noise schedule: {}'.format(noise_schedule))
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)

    self.register_buffer('betas', betas)
    self.register_buffer('alphas', alphas)
    self.register_buffer('alpha_cumprod', alpha_cumprod)
    self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)
    self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
    self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))

  def auxiliary_weight(self, t):
    if self.aux_loss_mode == 'all':
      return torch.ones((t.shape[0],), device=t.device, dtype=torch.float32)
    if self.aux_loss_mode == 'small_t':
      max_t = int(max(0, min(1.0, self.aux_t_max_ratio)) * (self.timesteps - 1))
      return (t <= max_t).float()
    if self.aux_loss_mode == 'snr':
      ac = self.alpha_cumprod.gather(0, t).float()
      snr = ac / torch.clamp(1.0 - ac, min=1e-8)
      # Weight is close to 1 for high-SNR/small-t samples and reduced for noisy large-t samples.
      return torch.clamp(snr / max(self.aux_snr_gamma, 1e-8), max=1.0)
    raise ValueError('Unsupported auxiliary loss mode: {}'.format(self.aux_loss_mode))

  def q_sample(self, z0, t, noise):
    return extract(self.sqrt_alpha_cumprod, t, z0.shape) * z0 + extract(self.sqrt_one_minus_alpha_cumprod, t, z0.shape) * noise

  def predict_z0(self, z_t, t, noise_pred):
    return (z_t - extract(self.sqrt_one_minus_alpha_cumprod, t, z_t.shape) * noise_pred) / extract(self.sqrt_alpha_cumprod, t, z_t.shape)

  def training_losses(self, ct, xray1, xray2):
    # ct [B, D, H, W] -> [B, 1, D, H, W]
    ct_3d = ct.unsqueeze(1)
    z0 = self.autoencoder.encode(ct_3d)
    t = torch.randint(0, self.timesteps, (ct.shape[0],), device=ct.device, dtype=torch.long)
    noise = torch.randn_like(z0)
    z_t = self.q_sample(z0, t, noise)

    cond = self.cond_encoder(xray1, xray2, z0.shape[-3:])
    if self.condition_refiner is not None:
      cond = self.condition_refiner(cond, self.alpha_cumprod.gather(0, t))
    if self.cond_drop_prob > 0:
      keep_mask = (torch.rand((ct.shape[0], 1, 1, 1, 1), device=ct.device) > self.cond_drop_prob).float()
      cond = cond * keep_mask
    noise_pred = self.denoiser(z_t, cond, t)
    z0_pred = self.predict_z0(z_t, t, noise_pred)

    ct_pred = self.autoencoder.decode(z0_pred).squeeze(1)
    ct_ae = self.autoencoder.decode(z0).squeeze(1)
    losses = {
      'diffusion': F.mse_loss(noise_pred, noise),
      'recon': torch.abs(ct_pred - ct).flatten(1).mean(1),
      'ae': torch.abs(ct_ae - ct).flatten(1).mean(1),
      'aux_weight': self.auxiliary_weight(t),
    }
    return losses, ct_pred

  def forward(self, ct, xray1, xray2):
    return self.training_losses(ct, xray1, xray2)

  @torch.no_grad()
  def ddim_sample(self, xray1, xray2, latent_shape, sample_steps=50, eta=0.0, cfg_scale=1.0, force_uncond=False):
    b = xray1.shape[0]
    z_t = torch.randn((b, self.latent_channels, latent_shape[0], latent_shape[1], latent_shape[2]), device=xray1.device)
    cond = self.cond_encoder(xray1, xray2, latent_shape)
    uncond = torch.zeros_like(cond)
    detail_components = (self.condition_refiner.components(cond)
                         if self.condition_refiner is not None and not force_uncond else None)

    sample_steps = int(sample_steps)
    if not 1 <= sample_steps <= self.timesteps:
      raise ValueError('sample_steps must be in [1, timesteps]')
    step_indices = torch.linspace(self.timesteps - 1, 0, sample_steps, device=xray1.device).long()

    for i in range(len(step_indices)):
      t_cur = step_indices[i]
      t = torch.full((b,), int(t_cur.item()), device=xray1.device, dtype=torch.long)
      step_cond = (self.condition_refiner.combine(cond, detail_components, self.alpha_cumprod.gather(0, t))
                   if detail_components is not None else cond)
      if force_uncond:
        noise_pred = self.denoiser(z_t, uncond, t)
      elif cfg_scale is not None and cfg_scale != 1.0:
        noise_cond = self.denoiser(z_t, step_cond, t)
        noise_uncond = self.denoiser(z_t, uncond, t)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
      else:
        noise_pred = self.denoiser(z_t, step_cond, t)
      a_t = extract(self.alpha_cumprod, t, z_t.shape)
      sqrt_a_t = torch.sqrt(a_t)
      sqrt_1m_a_t = torch.sqrt(1.0 - a_t)
      z0_pred = (z_t - sqrt_1m_a_t * noise_pred) / sqrt_a_t

      if i == len(step_indices) - 1:
        z_t = z0_pred
        continue

      t_prev = torch.full((b,), int(step_indices[i + 1].item()), device=xray1.device, dtype=torch.long)
      a_prev = extract(self.alpha_cumprod, t_prev, z_t.shape)
      sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t)) * torch.sqrt(1.0 - a_t / a_prev)
      c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0))
      noise = torch.randn_like(z_t) if eta != 0 else torch.zeros_like(z_t)
      z_t = torch.sqrt(a_prev) * z0_pred + c * noise_pred + sigma * noise

    return self.autoencoder.decode(z_t).squeeze(1)
