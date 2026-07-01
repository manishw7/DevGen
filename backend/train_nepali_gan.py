"""
DevGen Framework — Rectangular Word GAN (128x64)
Features: ResNet+SN, EMA, Hinge Loss, Checkpoint Resumption
Designed specifically for training on words (INDIC dataset) in Kaggle.
"""

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

LATENT_DIM = 128
IMG_H = 64
IMG_W = 128
CHANNELS = 1

# ── DiffAugment (Safe for Handwriting) ─────────────────────────────────────
# We only use translation. Cutout and Color destroy character strokes.

def DiffAugment(x, policy="translation"):
    if not policy:
        return x
    if "translation" in policy:
        ratio = 0.125
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
        x = xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    return x.contiguous()


# ── Architecture ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with spectral normalization."""
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(ic, oc, 3, stride, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(oc, oc, 3, 1, 1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sc = nn.Sequential() if stride == 1 and ic == oc else nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ic, oc, 1, stride, 0))
        )

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))) + self.sc(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_h = IMG_H // 16  # 4
        self.init_w = IMG_W // 16  # 8
        self.l1 = nn.Linear(LATENT_DIM, 512 * self.init_h * self.init_w)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2), ResBlock(512, 256),  # 4x8 → 8x16
            nn.Upsample(scale_factor=2), ResBlock(256, 128),  # 8x16 → 16x32
            nn.Upsample(scale_factor=2), ResBlock(128, 64),   # 16x32 → 32x64
            nn.Upsample(scale_factor=2), ResBlock(64, 32),    # 32x64 → 64x128
            nn.Conv2d(32, CHANNELS, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(-1, 512, self.init_h, self.init_w)
        return self.main(out)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            ResBlock(CHANNELS, 64, stride=2),   # 64x128 → 32x64
            ResBlock(64, 128, stride=2),        # 32x64 → 16x32
            ResBlock(128, 256, stride=2),       # 16x32 → 8x16
            ResBlock(256, 512, stride=2),       # 8x16 → 4x8
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(512 * (IMG_H // 16) * (IMG_W // 16), 1))
        )

    def forward(self, x):
        return self.main(x)


# ── Exponential Moving Average (EMA) ───────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)


# ── Training Loop ───────────────────────────────────────────────────────────

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[DevGen GAN] Training on {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    ema = EMA(G, decay=0.999)

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))

    start_step = 0
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")

    # Resume training if checkpoint exists
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} to resume training...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        ema.model.load_state_dict(ckpt["ema"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_step = ckpt["step"]
        print(f"Resumed successfully from step {start_step}")

    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.Grayscale(num_output_channels=CHANNELS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    fixed_z = torch.randn(16, LATENT_DIM, device=device)
    step = start_step
    n_critic = 2
    
    # Calculate time limit
    max_time_seconds = args.max_time_hours * 3600
    start_time = time.time()
    
    print(f"Training for up to {args.max_time_hours} hours...")

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            print(f"\n⏰ Time limit reached ({elapsed/3600:.2f}h). Stopping safely.")
            break

        for imgs, _ in loader:
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                break
                
            real = imgs.to(device)
            bs = real.size(0)

            # ── Train Discriminator ──
            for _ in range(n_critic):
                z = torch.randn(bs, LATENT_DIM, device=device)
                with torch.no_grad():
                    fake = G(z)

                # Safe DiffAugment (only translation)
                d_real = D(DiffAugment(real, policy="translation"))
                d_fake = D(DiffAugment(fake, policy="translation"))
                
                loss_D = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

            # ── Train Generator ──
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = G(z)
            loss_G = -D(DiffAugment(fake, policy="translation")).mean()
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            ema.update(G)
            step += 1

            if step % args.log_interval == 0:
                rate = step / max(1, elapsed)
                print(f"Step {step} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f} | {rate:.1f} steps/s")

            if step % args.sample_interval == 0:
                with torch.no_grad():
                    s = ema.model(fixed_z)
                save_image(s, os.path.join(args.output_dir, f"samples/step_{step}.png"), nrow=4, normalize=True)

            if step % args.save_interval == 0:
                torch.save(ema.model.state_dict(), os.path.join(args.output_dir, "generator_ema.pth"))
                torch.save(G.state_dict(), os.path.join(args.output_dir, "generator.pth"))
                torch.save({
                    "step": step,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "ema": ema.model.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                }, ckpt_path)

    # Final save
    torch.save(ema.model.state_dict(), os.path.join(args.output_dir, "generator_ema.pth"))
    torch.save(G.state_dict(), os.path.join(args.output_dir, "generator.pth"))
    print(f"[DevGen GAN] Training session finished at step {step}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DevGen GAN Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--sample-interval", type=int, default=1000, help="Save samples every N steps")
    parser.add_argument("--save-interval", type=int, default=5000, help="Save models every N steps")
    parser.add_argument("--log-interval", type=int, default=100, help="Log losses every N steps")
    parser.add_argument("--output-dir", type=str, default="gan_outputs", help="Output directory")
    parser.add_argument("--max-time-hours", type=float, default=10.0, help="Max training time before safe exit")
    args = parser.parse_args()
    train(args)
