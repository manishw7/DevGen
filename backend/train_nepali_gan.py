"""
DevGen Framework — GAN Training Script (v3)
Generates synthetic Devanagari handwritten character/word images.

Architecture: ResNet GAN with DiffAugment, Hinge Loss, Spectral Normalization,
              and Exponential Moving Average (EMA) for high-quality generation.

Inspired by Chhatkuli et al. (2021) and enhanced with modern GAN techniques
from BigGAN (Brock et al. 2019) and DiffAugment (Zhao et al. 2020).

Usage (local):
    python -m backend.train_nepali_gan \\
        --data-dir data/Images/Images \\
        --steps 100000 --batch-size 64

Usage (Kaggle): See DevGen_GAN_Notebook.ipynb
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ── Hyperparameters ─────────────────────────────────────────────────────────
LATENT_DIM = 128
IMG_SIZE = 64
CHANNELS = 1
G_CH = 64  # Base channel multiplier for Generator
D_CH = 64  # Base channel multiplier for Discriminator


# ── DiffAugment (Zhao et al. 2020) ─────────────────────────────────────────

def DiffAugment(x, policy="color,translation,cutout"):
    """Apply differentiable augmentations to stabilize GAN training."""
    if not policy:
        return x
    for p in policy.split(","):
        for fn in _AUGMENT_FNS[p]:
            x = fn(x)
    return x.contiguous()


def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def _rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.5 + 0.5) + mean


def _rand_translation(x, ratio=0.125):
    sx = int(x.size(2) * ratio + 0.5)
    sy = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device), indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    xp = F.pad(x, [1, 1, 1, 1])
    return xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _rand_cutout(x, ratio=0.5):
    cs = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cs[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cs[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(cs[0], device=x.device),
        torch.arange(cs[1], device=x.device), indexing="ij",
    )
    gx = torch.clamp(gx + ox - cs[0] // 2, 0, x.size(2) - 1)
    gy = torch.clamp(gy + oy - cs[1] // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


_AUGMENT_FNS = {
    "color": [_rand_brightness, _rand_saturation, _rand_contrast],
    "translation": [_rand_translation],
    "cutout": [_rand_cutout],
}


# ── Model Architecture ─────────────────────────────────────────────────────

class ResBlockUp(nn.Module):
    """Generator residual block with bilinear upsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(x))
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h)))
        sc = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return h + self.shortcut(sc)


class ResBlockDown(nn.Module):
    """Discriminator residual block with average-pool downsampling."""

    def __init__(self, in_ch, out_ch, first_block=False):
        super().__init__()
        self.first_block = first_block
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.shortcut = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = x if self.first_block else F.relu(x)
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        sc = self.shortcut(x)
        return F.avg_pool2d(h + sc, 2)


class Generator(nn.Module):
    """ResNet Generator: z(128) → 4×4 → 8 → 16 → 32 → 64 grayscale image."""

    def __init__(self, z_dim=LATENT_DIM, ch=G_CH):
        super().__init__()
        self.linear = nn.Linear(z_dim, (ch * 8) * 4 * 4)
        self.blocks = nn.ModuleList([
            ResBlockUp(ch * 8, ch * 4),   # 4 → 8
            ResBlockUp(ch * 4, ch * 2),   # 8 → 16
            ResBlockUp(ch * 2, ch),       # 16 → 32
            ResBlockUp(ch, ch),           # 32 → 64
        ])
        self.out = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, CHANNELS, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.linear(z).view(-1, G_CH * 8, 4, 4)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class Discriminator(nn.Module):
    """ResNet Discriminator: 64×64 → scalar validity score."""

    def __init__(self, ch=D_CH):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlockDown(CHANNELS, ch, first_block=True),  # 64 → 32
            ResBlockDown(ch, ch * 2),                      # 32 → 16
            ResBlockDown(ch * 2, ch * 4),                  # 16 → 8
            ResBlockDown(ch * 4, ch * 8),                  # 8 → 4
        ])
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ch * 8, 1)),
        )

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return self.out(h)


# ── Exponential Moving Average (EMA) ───────────────────────────────────────

class EMA:
    """Maintains an exponential moving average of model parameters for
    smoother, higher-quality generation at inference time."""

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)


# ── Training Loop ───────────────────────────────────────────────────────────

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[DevGen GAN v3] Training on {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    ema = EMA(G, decay=0.999)

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=CHANNELS),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    fixed_z = torch.randn(16, LATENT_DIM, device=device)
    step = 0
    n_critic = 2  # Train D twice per G step
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        for imgs, _ in loader:
            if step >= args.steps:
                break
            real = imgs.to(device)
            bs = real.size(0)

            # ── Train Discriminator ──
            for _ in range(n_critic):
                z = torch.randn(bs, LATENT_DIM, device=device)
                with torch.no_grad():
                    fake = G(z)
                d_real = D(DiffAugment(real))
                d_fake = D(DiffAugment(fake))
                loss_D = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ── Train Generator ──
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = G(z)
            loss_G = -D(DiffAugment(fake)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # Update EMA
            ema.update(G)

            step += 1
            pbar.update(1)

            if step % args.log_interval == 0:
                pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.3f}")

            if step % args.sample_interval == 0:
                with torch.no_grad():
                    ema_samples = ema.model(fixed_z)
                save_image(ema_samples, os.path.join(args.output_dir, f"samples/step_{step}.png"),
                           nrow=4, normalize=True)
                torch.save(ema.model.state_dict(), os.path.join(args.output_dir, "generator_ema.pth"))
                torch.save(G.state_dict(), os.path.join(args.output_dir, "generator.pth"))

    pbar.close()
    print(f"[DevGen GAN v3] Training complete. Models saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DevGen GAN v3 Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data (ImageFolder)")
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--sample-interval", type=int, default=2000, help="Save samples every N steps")
    parser.add_argument("--log-interval", type=int, default=100, help="Log losses every N steps")
    parser.add_argument("--output-dir", type=str, default="gan_outputs", help="Output directory")
    args = parser.parse_args()
    train(args)
