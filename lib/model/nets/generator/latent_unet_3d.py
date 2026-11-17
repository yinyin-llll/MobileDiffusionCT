"""Time-conditioned volumetric U-Net used only in the denoiser ablation."""
import torch
from torch import nn
from torch.nn import functional as F


class LatentUNetDenoiser3D(nn.Module):
    """Two spatial reductions, skip concatenation, and two learned upsamplers.

    The entry width and time embedding match the residual denoiser. Widths
    are h, 2h, 2h; increased capacity is reported, not parameter matched.
    """
    def __init__(self, latent_channels=128, cond_channels=128,
                 hidden_channels=256, time_dim=256):
        super().__init__()
        from .latent_diffusion_3d import ResidualBlock3D
        h = hidden_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        self.in_proj = nn.Conv3d(latent_channels + cond_channels, h, 3, padding=1)
        self.encoder0 = ResidualBlock3D(h, time_dim)
        self.down0 = nn.Conv3d(h, 2*h, 3, stride=2, padding=1)
        self.encoder1 = ResidualBlock3D(2*h, time_dim)
        self.down1 = nn.Conv3d(2*h, 2*h, 3, stride=2, padding=1)
        self.bottleneck = ResidualBlock3D(2*h, time_dim)
        self.up1 = nn.ConvTranspose3d(2*h, 2*h, 4, stride=2, padding=1)
        self.fuse1 = nn.Conv3d(4*h, 2*h, 1)
        self.decoder1 = ResidualBlock3D(2*h, time_dim)
        self.up0 = nn.ConvTranspose3d(2*h, h, 4, stride=2, padding=1)
        self.fuse0 = nn.Conv3d(2*h, h, 1)
        self.decoder0 = ResidualBlock3D(h, time_dim)
        self.out_norm = nn.GroupNorm(8, h)
        self.out_proj = nn.Conv3d(h, latent_channels, 3, padding=1)

    def forward(self, z_t, cond, t):
        from .latent_diffusion_3d import timestep_embedding
        if z_t.shape[0] != cond.shape[0] or z_t.shape[-3:] != cond.shape[-3:]:
            raise ValueError('Noisy latent and condition grids must match')
        if any(n < 4 or n % 4 for n in z_t.shape[-3:]):
            raise ValueError('U-Net latent dimensions must be multiples of four')
        time = self.time_mlp(timestep_embedding(t, self.time_dim))
        skip0 = self.encoder0(self.in_proj(torch.cat((z_t, cond), dim=1)), time)
        skip1 = self.encoder1(self.down0(skip0), time)
        h = self.bottleneck(self.down1(skip1), time)
        h = self.decoder1(self.fuse1(torch.cat((self.up1(h), skip1), dim=1)), time)
        h = self.decoder0(self.fuse0(torch.cat((self.up0(h), skip0), dim=1)), time)
        return self.out_proj(F.silu(self.out_norm(h)))
