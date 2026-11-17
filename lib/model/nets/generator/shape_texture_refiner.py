"""Shape-guided, diffusion-time-gated residual refinement of 3D conditions.

The module acts on learned 3D features, not anatomical masks or calibrated CT
edges. It uses no extra supervision. Shape/texture decomposition and edge
gating are established ideas; this is an experimental integration with the
project's multistage adaptive lifting, not a claim of first use.
"""
import torch
from torch import nn
from torch.nn import functional as F


def box_blur3d(features, kernel):
    radius = kernel // 2
    return F.avg_pool3d(F.pad(features, (radius,) * 6, mode='replicate'), kernel, stride=1)


def feature_differences3d(features):
    """Centered differences with replicated boundaries, retaining D/H/W size."""
    padded = F.pad(features, (1,) * 6, mode='replicate')
    dz = (padded[:, :, 2:, 1:-1, 1:-1] - padded[:, :, :-2, 1:-1, 1:-1]) * .5
    dy = (padded[:, :, 1:-1, 2:, 1:-1] - padded[:, :, 1:-1, :-2, 1:-1]) * .5
    dx = (padded[:, :, 1:-1, 1:-1, 2:] - padded[:, :, 1:-1, 1:-1, :-2]) * .5
    return dz, dy, dx


class ShapeContext3D(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.reduce = nn.Conv3d(channels, hidden, 1, bias=False)
        self.axes = nn.ModuleList([
            nn.Conv3d(hidden, hidden, kernel, padding=padding, groups=hidden,
                      bias=False, padding_mode='replicate')
            for kernel, padding in [((5, 1, 1), (2, 0, 0)),
                                    ((1, 5, 1), (0, 2, 0)),
                                    ((1, 1, 5), (0, 0, 2))]])
        self.out = nn.Sequential(nn.GroupNorm(8, hidden, affine=False), nn.SiLU(),
                                 nn.Conv3d(hidden, channels, 1, bias=False))

    def forward(self, features):
        coarse = self.reduce(box_blur3d(features, 5))
        return self.out(sum(axis(coarse) for axis in self.axes) / len(self.axes))


class ShapeTextureConditionRefiner(nn.Module):
    def __init__(self, channels, mode='both'):
        super().__init__()
        if mode not in ('shape', 'texture', 'both'):
            raise ValueError('Unknown condition detail mode: ' + mode)
        self.mode = mode
        self.use_shape, self.use_texture = mode != 'texture', mode != 'shape'
        hidden = max(8, (channels // 4 // 8) * 8)
        if self.use_shape:
            self.shape_context = ShapeContext3D(channels, hidden)
            self.shape_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), .5))
        if self.use_texture:
            # High-pass residual and the three directional feature derivatives.
            # No biases/normalization offsets: a constant feature has zero
            # texture correction, including at replicated volume boundaries.
            self.texture_features = nn.Sequential(
                nn.Conv3d(4 * channels, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden, affine=False), nn.SiLU(),
                nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                nn.Conv3d(hidden, channels, 1, bias=False))
            self.texture_gate = nn.Conv3d(2 * channels + 1, channels, 1)
            self.texture_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), .5))

    def components(self, features):
        """Cache image-dependent branches once during iterative sampling."""
        shape = self.shape_context(features) if self.use_shape else None
        shape_delta = .1 * self.shape_scale.tanh() * shape if self.use_shape else None
        texture_delta = None
        if self.use_texture:
            differences = feature_differences3d(features)
            highpass = features - box_blur3d(features, 3)
            local = self.texture_features(torch.cat((highpass, *differences), dim=1))
            edge = sum(value.abs().mean(1, keepdim=True) for value in differences) / 3
            relative_edge = (edge / edge.mean((-3, -2, -1), keepdim=True).clamp_min(1e-6)).clamp(0, 4)
            context = shape if shape is not None else torch.zeros_like(features)
            gate = self.texture_gate(torch.cat((features, context, relative_edge), dim=1)).sigmoid()
            texture_delta = .1 * self.texture_scale.tanh() * gate * local
        return shape_delta, texture_delta

    @staticmethod
    def combine(features, components, alpha_bar):
        if alpha_bar.ndim != 1 or alpha_bar.shape[0] != features.shape[0]:
            raise ValueError('Expected one diffusion alpha_bar per batch item')
        if not torch.isfinite(alpha_bar).all() or torch.any((alpha_bar < 0) | (alpha_bar > 1)):
            raise ValueError('alpha_bar must be finite in [0,1]')
        shape_delta, texture_delta = components
        output = features if shape_delta is None else features + shape_delta
        if texture_delta is not None:
            # Coarse structure remains available at every t. Fine texture is
            # introduced progressively as the reverse process becomes cleaner.
            gate = alpha_bar.float().sqrt().reshape(-1, 1, 1, 1, 1).to(features.dtype)
            output = output + gate * texture_delta
        return output

    def forward(self, features, alpha_bar):
        return self.combine(features, self.components(features), alpha_bar)
