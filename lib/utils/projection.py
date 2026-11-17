"""Revision v2 parallel-beam geometry in normalized voxel coordinates.

CT: B,D,H,W. Detector: D,W. No display flip, no per-view normalization.
An angle rotates the sampling grid: xs=cos(a)*x-sin(a)*y,
ys=sin(a)*x+cos(a)*y. Projections average along detector ray axis H.
This is a voxel-space simulator, not a calibrated clinical X-ray model.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PROTOCOL = 'parallel_mean_v2'


def project_volume(ct, angles):
    if ct.ndim != 4 or ct.shape[-1] != ct.shape[-2]:
        raise ValueError('Expected B,D,H,W with square H,W')
    b, d, h, w = ct.shape
    slices = ct.reshape(b*d, 1, h, w)
    views = []
    for angle in angles:
        a = math.radians(float(angle))
        matrix = ct.new_tensor([[math.cos(a), -math.sin(a), 0.0],
                                [math.sin(a), math.cos(a), 0.0]])
        matrix = matrix.unsqueeze(0).expand(b*d, -1, -1)
        def integrate(source, transform):
            grid = F.affine_grid(transform, source.shape, align_corners=True)
            rotated = F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            return rotated.mean(dim=2)
        # Recompute grids in backward instead of storing V full rotated volumes.
        projected = checkpoint(integrate, slices, matrix, use_reentrant=False) if slices.requires_grad else integrate(slices, matrix)
        views.append(projected.reshape(b, d, w))
    return torch.stack(views, dim=1)


def uniform_angles(n, span=180.0, offset=0.0):
    if n < 1 or not 0 < span <= 360:
        raise ValueError('Invalid number of views or angular span')
    return [offset + span*i/n for i in range(n)]
