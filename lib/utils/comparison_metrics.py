"""Shared CT comparison protocol: pretrained slice LPIPS and inference MAC FLOPs.

FLOPs convention: 1 MAC = 2 FLOPs. Count executed convolutions (including 3D
and transposed convolutions), matrix products and fused attention. This is a
MAC-based estimate, excluding normalization, activation, interpolation and
elementwise/sampler operations. Run on the COMPLETE reconstruction, not one
denoising step. LPIPS and data loading must be outside the profiled callable.
"""
from collections import Counter
import math

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class VolumeLPIPS:
    """AlexNet LPIPS v0.1; average all slices in each of the three CT planes.

    Inputs are matching normalized [0, 1] D,H,W arrays. Gray is repeated into
    RGB; no case-specific rescaling, HU windows or resizing is applied.
    """

    def __init__(self, device='cpu', batch_size=16):
        import lpips
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('LPIPS batch size must be positive')
        self.model = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, truth, prediction):
        truth, prediction = np.asarray(truth), np.asarray(prediction)
        if truth.shape != prediction.shape or truth.ndim != 3:
            raise ValueError('LPIPS expects matching D,H,W volumes')
        for array in (truth, prediction):
            if not np.isfinite(array).all() or array.min() < -1e-5 or array.max() > 1.00001:
                raise ValueError('LPIPS inputs must be finite and normalized to [0,1]')
        plane_means = []
        for axis in range(3):
            a, b = np.moveaxis(truth, axis, 0), np.moveaxis(prediction, axis, 0)
            if min(a.shape[1:]) < 32:
                raise ValueError('AlexNet LPIPS needs slice dimensions >=32')
            distances = []
            for start in range(0, len(a), self.batch_size):
                def rgb(x):
                    x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
                    return x[:, None].repeat(1, 3, 1, 1).mul(2).sub(1)
                d = self.model(rgb(a[start:start+self.batch_size]), rgb(b[start:start+self.batch_size]))
                distances.extend(d.flatten().cpu().tolist())
            plane_means.append(float(np.mean(distances)))
        value = float(np.mean(plane_means))
        if not math.isfinite(value):
            raise ValueError('Nonfinite LPIPS')
        return value


class InferenceFlops(TorchDispatchMode):
    """Count actual executed ATen convolution/GEMM/attention MAC operations."""

    def __init__(self):
        super().__init__()
        self.by_operator = Counter()
        self.calls = Counter()
        self.other_operators = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func._schema.name
        flops = None
        if name in ('aten::convolution', 'aten::_convolution'):
            x, weight = args[:2]
            transposed = args[6] if len(args) > 6 else kwargs.get('transposed', False)
            kernel = math.prod(weight.shape[2:])
            flops = 2 * (x.numel() if transposed else out.numel()) * weight.shape[1] * kernel
        elif name in ('aten::conv1d', 'aten::conv2d', 'aten::conv3d',
                      'aten::conv_transpose1d', 'aten::conv_transpose2d', 'aten::conv_transpose3d'):
            x, weight = args[:2]
            flops = 2 * (x.numel() if 'transpose' in name else out.numel()) * weight.shape[1] * math.prod(weight.shape[2:])
        elif name == 'aten::linear':
            flops = 2 * out.numel() * args[0].shape[-1]
        elif name in ('aten::mm', 'aten::bmm', 'aten::addmm', 'aten::baddbmm'):
            offset = 1 if name in ('aten::addmm', 'aten::baddbmm') else 0
            a = args[offset]
            flops = 2 * out.numel() * a.shape[-1]
        elif name == 'aten::_native_multi_head_attention':
            q, k = args[:2]
            batch, length, channels = q.shape
            flops = 8 * batch * length * channels**2 + 4 * batch * length * k.shape[1] * channels
        elif (name.startswith('aten::_scaled_dot_product_') or name == 'aten::scaled_dot_product_attention') and 'backward' not in name:
            q, k, v = args[:3]
            flops = 2 * math.prod(q.shape[:-2]) * q.shape[-2] * k.shape[-2] * (q.shape[-1] + v.shape[-1])
        if flops is None:
            self.other_operators[name] += 1
        else:
            self.by_operator[name] += int(flops)
            self.calls[name] += 1
        return out

    def report(self):
        total = sum(self.by_operator.values())
        if total <= 0:
            raise RuntimeError('No supported FLOP operations executed')
        return dict(FLOPs=total, FLOPs_T=total/1e12, MACs=total/2,
                    convention='1 MAC = 2 FLOPs; full reconstruction at batch 1',
                    included='convolution 1D/2D/3D, transposed convolution, GEMM, attention GEMMs',
                    excluded='normalization, activations, interpolation, elementwise and sampler operations; loading and quality metrics',
                    by_operator=dict(self.by_operator), calls=dict(self.calls),
                    uncounted_operator_calls=dict(self.other_operators))


def profile_reconstruction(callable_):
    # Disable fused Transformer blocks so their constituent GEMMs are visible.
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad(), InferenceFlops() as counter:
            callable_()
        return counter.report()
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
