"""Coordinate-conditioned feature lifting with residual encoder-stage fusion.

This experimental feature operator uses known view directions as anchors; it
does not introduce a new X-ray acquisition model or claim calibrated inversion.
The adaptive path uses integer gathers and learned Gaussian neighborhood
weights, not grid_sample/bilinear interpolation.
"""
import math

import torch
from torch import nn


def normalized_kernel_sample(features, uv, shift, sigma):
    """Sample a 3x3 pixel neighborhood with normalized anisotropic weights.

    features: B,C,H,W; uv/shift/sigma: B,N,2. Shifts and widths use feature
    pixels. The support is anchored to the nominal projection, with a bounded
    subpixel shift; invalid pixels never wrap around image boundaries.
    """
    b, c, h, w = features.shape
    if uv.shape != shift.shape or uv.shape != sigma.shape or uv.shape[0] != b:
        raise ValueError('Expected matching B,N,2 coordinates, shifts, widths')
    if uv.shape[-1] != 2 or not torch.isfinite(uv).all():
        raise ValueError('Invalid detector coordinates')
    if not torch.isfinite(sigma).all() or torch.any(sigma <= 0):
        raise ValueError('Kernel widths must be finite and positive')
    scale = uv.new_tensor([max(w - 1, 0), max(h - 1, 0)]) / 2
    nominal = (uv + 1) * scale
    center = nominal + shift
    anchor = nominal.round().long()
    inside = (uv.abs() <= 1).all(dim=-1)
    pixels = features.flatten(2)
    numerator = features.new_zeros((b, c, uv.shape[1]))
    denominator = uv.new_zeros((b, 1, uv.shape[1]))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ix, iy = anchor[..., 0] + dx, anchor[..., 1] + dy
            valid = inside & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            delta = torch.stack((ix, iy), dim=-1).to(center.dtype) - center
            weight = torch.exp(-0.5 * ((delta / sigma) ** 2).sum(-1)) * valid
            index = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).unsqueeze(1)
            sample = pixels.gather(2, index.expand(-1, c, -1))
            numerator = numerator + sample * weight[:, None].to(features.dtype)
            denominator = denominator + weight[:, None]
    result = (numerator.float() / denominator.clamp_min(1e-8)).to(features.dtype)
    return result, inside


class AdaptiveRayFeatureLift(nn.Module):
    """Learn depth-dependent sampling, gain and per-voxel view reliability."""
    def __init__(self, channels):
        super().__init__()
        # Query: detector u/z, position along the ray, sin/cos view angle.
        self.coordinate_kernel = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 5))
        self.view_confidence = nn.Conv3d(channels, 1, 1)

    def forward(self, features, shape, angles, geometry=None):
        b, views, channels, _, _ = features.shape
        d, h, w = shape
        device = features.device
        if geometry is not None:
            if geometry.shape != (b, views, 3) or not torch.isfinite(geometry).all():
                raise ValueError('Expected finite B,V,3 observation geometry')
            if torch.any((geometry[..., 0] != 0) & (geometry[..., 0] != 1)):
                raise ValueError('Observation kind must be projection=0 or axial slice=1')
            if torch.any((geometry[..., 0] == 1) & (geometry[..., 2].abs() > 1)):
                raise ValueError('Slice outside normalized CT grid')
        # Coordinates and exponentials remain float32 under mixed precision.
        z, y, x = [torch.linspace(-1, 1, n, device=device) for n in shape]
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        lifted, logits, masks = [], [], []
        for i in range(views):
            theta = angles[:, i].float() * (math.pi / 180)
            cs, sn = theta.cos()[:, None, None, None], theta.sin()[:, None, None, None]
            u = xx[None] * cs + yy[None] * sn
            ray_depth = -xx[None] * sn + yy[None] * cs
            vertical = zz[None].expand(b, -1, -1, -1)
            plane_support = None
            if geometry is not None and torch.any(geometry[:, i, 0] == 1):
                is_slice = (geometry[:, i, 0] == 1)[:, None, None, None]
                distance = vertical - geometry[:, i, 2, None, None, None]
                # Axial slices observe x/y on a known z plane. Never interpret
                # a slice index as a projection angle or backproject a slice.
                u = torch.where(is_slice, xx[None].expand_as(u), u)
                vertical = torch.where(is_slice, yy[None].expand_as(vertical), vertical)
                ray_depth = torch.where(is_slice, distance, ray_depth)
                plane_support = torch.where(is_slice,
                    torch.exp(-0.5 * (distance / (2. / max(d-1, 1))) ** 2),
                    torch.ones_like(distance))
            query = torch.stack((u, vertical, ray_depth,
                                 sn.expand_as(u), cs.expand_as(u)), dim=-1).reshape(b, -1, 5)
            with torch.autocast(device_type=device.type, enabled=False):
                params = self.coordinate_kernel(query.float())
                shift = 0.5 * params[..., :2].tanh()
                sigma = 0.35 + 1.15 * params[..., 2:4].sigmoid()
                depth_logit = params[..., 4]
            uv = torch.stack((u, vertical), dim=-1).reshape(b, -1, 2)
            sample, valid = normalized_kernel_sample(features[:, i], uv, shift, sigma)
            gain = 0.5 + depth_logit.sigmoid()
            volume = (sample * gain[:, None].to(sample.dtype)).reshape(b, channels, d, h, w)
            if plane_support is not None:
                volume = volume * plane_support[:, None].to(volume.dtype)
            confidence = self.view_confidence(volume).float() + depth_logit.reshape(b, 1, d, h, w)
            if plane_support is not None:
                confidence = confidence + plane_support[:, None].clamp_min(1e-8).log()
            lifted.append(volume)
            logits.append(confidence)
            masks.append(valid.reshape(b, 1, d, h, w))
        mask = torch.stack(masks, dim=1)
        weights = torch.stack(logits, dim=1).masked_fill(~mask, -1e4).softmax(dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (torch.stack(lifted, dim=1) * weights.to(features.dtype)).sum(dim=1)


class ResidualStageFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv3d(2 * channels, channels, 1)
        self.residual = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, 1))

    def forward(self, previous, evidence):
        gate = self.gate(torch.cat((previous, evidence), dim=1)).sigmoid()
        return previous + gate * self.residual(evidence)


class ResidualPyramidConditionEncoder(nn.Module):
    """Use both convolution stages and every ViT block as 3D skip evidence."""
    def __init__(self, backbone, channels, mode):
        super().__init__()
        if mode not in ('adaptive_last', 'pyramid_parallel', 'adaptive_pyramid'):
            raise ValueError('Unknown feature lifting mode: ' + mode)
        self.backbone = backbone
        self.mode = mode
        self.multiscale = mode != 'adaptive_last'
        self.adaptive = mode != 'pyramid_parallel'
        widths = [backbone.stem1[0].out_channels, channels] + [channels] * len(backbone.vit1)
        if not self.multiscale:
            widths = widths[-1:]
        self.stage_projections = nn.ModuleList([
            nn.Conv2d(c, channels, 1) if c != channels else nn.Identity() for c in widths])
        self.stage_lifts = nn.ModuleList([AdaptiveRayFeatureLift(channels) for _ in widths]) if self.adaptive else nn.ModuleList()
        self.stage_fusions = nn.ModuleList([ResidualStageFusion(channels) for _ in widths[1:]])
        self.stage_names = ['conv_stage_1', 'conv_stage_2'] + [f'vit_block_{i+1}' for i in range(len(backbone.vit1))]
        if not self.multiscale:
            self.stage_names = self.stage_names[-1:]

    def forward(self, xray1, xray2, target_shape):
        from .latent_diffusion_3d import lift_observation_features
        if xray2 is not None or xray1.ndim != 5:
            raise ValueError('Experimental lifting requires explicit B,V,C,H,W inputs')
        b, views, c, h, w = xray1.shape
        geometry = getattr(self, 'input_geometry', None)
        if geometry is None:
            angles = (torch.arange(views, device=xray1.device, dtype=torch.float32)
                      * self.backbone.backproject_angle_span / views + self.backbone.backproject_angle_offset)
            angles = angles[None].expand(b, -1)
            geometry = torch.zeros(b, views, 3, device=xray1.device)
            geometry[:, :, 1] = angles
        else:
            if geometry.shape != (b, views, 3) or not torch.isfinite(geometry).all():
                raise ValueError('Expected finite B,V,3 observation geometry')
            if torch.any((geometry[:, :, 0] != 0) & (geometry[:, :, 0] != 1)):
                raise ValueError('Unknown observation kind')
            angles = geometry[:, :, 1]
        features, stages = xray1.reshape(b * views, c, h, w), []
        for index, layer in enumerate(self.backbone.stem1):
            features = layer(features)
            if index in (2, 5):
                stages.append(features)
        for block in self.backbone.vit1:
            features = block(features)
            stages.append(features)
        if not self.multiscale:
            stages = stages[-1:]
        running = None
        for index, (stage, projection) in enumerate(zip(stages, self.stage_projections)):
            projected = projection(stage)
            fv = projected.reshape(b, views, *projected.shape[1:])
            slice_geometry = geometry if torch.any(geometry[:, :, 0] == 1) else None
            evidence = (self.stage_lifts[index](fv, target_shape, angles, geometry=slice_geometry) if self.adaptive
                        else lift_observation_features(fv, target_shape, geometry))
            running = evidence if running is None else self.stage_fusions[index - 1](running, evidence)
        return running
