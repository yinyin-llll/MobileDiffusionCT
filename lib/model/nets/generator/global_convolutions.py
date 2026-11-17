"""Opt-in whole-model convolution policies, including CT decoder upsampling."""
from torch import nn


class DepthwiseSeparableTranspose3D(nn.Module):
    """Depthwise transposed spatial filtering, then a dense 1x1 channel mixer."""
    def __init__(self, conv):
        super().__init__()
        if not isinstance(conv, nn.ConvTranspose3d) or conv.groups != 1:
            raise ValueError('Expected a dense ConvTranspose3d')
        self.depthwise = nn.ConvTranspose3d(
            conv.in_channels, conv.in_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding,
            output_padding=conv.output_padding, groups=conv.in_channels,
            bias=False, dilation=conv.dilation, padding_mode=conv.padding_mode)
        self.pointwise = nn.Conv3d(conv.in_channels, conv.out_channels, 1,
                                   bias=conv.bias is not None)
        self.to(device=conv.weight.device, dtype=conv.weight.dtype)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def apply_global_convolution_policy(module, policy):
    """Convert spatial kernels only; dense 1x1 channel projections are essential.

    Called after all condition/fusion modules are constructed and before weight
    initialization. Native depthwise filters already followed by pointwise mixing
    are retained in all_separable; all_standard expands them to groups=1.
    """
    from .latent_diffusion_3d import DepthwiseSeparableConv
    if policy not in ('all_separable', 'all_standard'):
        raise ValueError('Unknown global convolution policy: ' + policy)
    counts = dict(spatial_convolutions_replaced=0, transposed_convolutions_replaced=0,
                  native_depthwise_filters_expanded=0)
    for name, child in list(module.named_children()):
        spatial = isinstance(child, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)) and any(k > 1 for k in child.kernel_size)
        if spatial and policy == 'all_separable' and child.groups == 1:
            if isinstance(child, nn.ConvTranspose3d):
                replacement = DepthwiseSeparableTranspose3D(child)
                counts['transposed_convolutions_replaced'] += 1
            else:
                replacement = DepthwiseSeparableConv(child)
                counts['spatial_convolutions_replaced'] += 1
            setattr(module, name, replacement)
        elif spatial and policy == 'all_standard' and child.groups != 1:
            kwargs = dict(stride=child.stride, padding=child.padding,
                          dilation=child.dilation, groups=1,
                          bias=child.bias is not None, padding_mode=child.padding_mode)
            if isinstance(child, nn.ConvTranspose3d):
                kwargs['output_padding'] = child.output_padding
            replacement = type(child)(child.in_channels, child.out_channels, child.kernel_size, **kwargs)
            replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, replacement)
            counts['native_depthwise_filters_expanded'] += 1
        else:
            nested = apply_global_convolution_policy(child, policy)
            for key in counts:
                counts[key] += nested[key]
    return counts
