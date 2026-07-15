"""
Train a compact latent diffusion model FROM SCRATCH for Devanagari
handwritten word generation (Paper 2 v2). WordStylist-style recipe,
script-aware conditioning.

Design:
- Frozen SD VAE (stabilityai/sd-vae-ft-mse) maps 64x256 word images to
  4x8x32 latents. Latents are precomputed ONCE and cached (~150MB for 70k
  images), so training steps touch only the small UNet -> a full from-scratch
  run fits in a single Kaggle T4 session.
- UNet2DConditionModel (~65M params) with cross-attention over learned
  content-token embeddings (codepoint or akshara mode — the ablation).
- Classifier-free guidance: 10% of batches get the NULL condition.
- EMA weights, fp16 autocast, checkpoint/auto-resume every epoch.

Usage:
    python -m paper2.ldm.train_ldm --mode akshara --parquet-dir data/iiit_hindi_parquet \
        --output-dir paper2/runs/ldm_akshara --epochs 300
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper2.ldm.tokenizer import NULL_ID, PAD_ID, DevanagariTokenizer  # noqa: E402

IMG_H, IMG_W = 64, 256
LATENT_C, LATENT_H, LATENT_W = 4, IMG_H // 8, IMG_W // 8
VAE_SCALE = 0.18215
COND_DIM = 384


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_text(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", str(text).strip())


def prepare_word_image(image: Image.Image) -> Image.Image:
    """Grayscale word strip -> white-padded RGB 64x256, ink-preserving."""
    img = image.convert("L")
    w, h = img.size
    scale = IMG_H / h
    new_w = max(8, min(int(w * scale), IMG_W))
    img = img.resize((new_w, IMG_H), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (IMG_W, IMG_H), color=255)
    canvas.paste(img, ((IMG_W - new_w) // 2, 0))
    return canvas.convert("RGB")


class ContentEncoder(nn.Module):
    """Token embeddings + positions + a light transformer -> cross-attn memory."""

    def __init__(self, vocab_size: int, max_len: int, dim: int = COND_DIM):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=6, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(token_ids) + self.pos_emb[:, : token_ids.shape[1]]
        pad_mask = token_ids == PAD_ID
        return self.encoder(x, src_key_padding_mask=pad_mask)


def build_unet() -> "UNet2DConditionModel":
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        in_channels=LATENT_C,
        out_channels=LATENT_C,
        block_out_channels=(128, 256, 512),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=2,
        cross_attention_dim=COND_DIM,
        attention_head_dim=8,
        norm_num_groups=32,
    )


def precompute_latents(parquet_dir: str, split_prefix: str, cache_path: Path,
                       tokenizer: DevanagariTokenizer, device: str,
                       batch_size: int = 64, limit: int | None = None):
    """VAE-encode the whole split once; cache latents + token ids to disk."""
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu")
        # token ids depend on tokenizer mode — verify match
        if blob.get("tokenizer_mode") == tokenizer.mode and blob.get("max_len") == tokenizer.max_len:
            print(f"[latents] Reusing cache {cache_path} ({blob['latents'].shape[0]} samples)")
            return blob["latents"], blob["token_ids"]
        print("[latents] Cache tokenizer mismatch — rebuilding token ids only")
        latents = blob["latents"]
        texts = blob["texts"]
        token_ids = torch.tensor([tokenizer.encode(t) for t in texts], dtype=torch.long)
        torch.save({"latents": latents, "token_ids": token_ids, "texts": texts,
                    "tokenizer_mode": tokenizer.mode, "max_len": tokenizer.max_len}, cache_path)
        return latents, token_ids

    from datasets import load_dataset
    from diffusers import AutoencoderKL

    files = sorted(str(p) for p in Path(parquet_dir).glob(f"{split_prefix}-*.parquet"))
    spec = f"train[:{limit}]" if limit else "train"
    dataset = load_dataset("parquet", data_files=files, split=spec)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    if device == "cuda":
        vae = vae.half()

    all_latents, texts = [], []
    batch_imgs: list[torch.Tensor] = []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        if device == "cuda":
            x = x.half()
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * VAE_SCALE
        all_latents.append(lat.float().cpu())
        batch_imgs.clear()

    for i, sample in enumerate(dataset):
        img = sample["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        img = prepare_word_image(img)
        arr = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        batch_imgs.append(arr)
        texts.append(normalize_text(sample["text"]))
        if len(batch_imgs) == batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"[latents] {i + 1}/{len(dataset)}")
    flush()

    latents = torch.cat(all_latents)
    token_ids = torch.tensor([tokenizer.encode(t) for t in texts], dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "token_ids": token_ids, "texts": texts,
                "tokenizer_mode": tokenizer.mode, "max_len": tokenizer.max_len}, cache_path)
    print(f"[latents] Cached {latents.shape[0]} latents -> {cache_path} "
          f"({cache_path.stat().st_size / 1e6:.0f} MB)")
    del vae
    if device == "cuda":
        torch.cuda.empty_cache()
    return latents, token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Devanagari handwriting LDM from scratch.")
    parser.add_argument("--mode", choices=["codepoint", "akshara"], required=True)
    parser.add_argument("--parquet-dir", default="data/iiit_hindi_parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--save-every", type=int, default=10, help="Epochs between checkpoints.")
    parser.add_argument("--preview-every", type=int, default=25, help="Epochs between sample grids.")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--latent-cache", default=None, help="Defaults to <output-dir>/../latents_train.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer (built once from the train split, cached) ----
    tok_path = out_dir / "tokenizer.json"
    if tok_path.exists():
        tokenizer = DevanagariTokenizer.load(tok_path)
    else:
        from datasets import load_dataset
        files = sorted(str(p) for p in Path(args.parquet_dir).glob("train-*.parquet"))
        spec = f"train[:{args.train_limit}]" if args.train_limit else "train"
        texts = [normalize_text(t) for t in load_dataset("parquet", data_files=files, split=spec)["text"]]
        tokenizer = DevanagariTokenizer.build(texts, mode=args.mode)
        tokenizer.save(tok_path)
    print(f"[train] mode={args.mode} vocab={tokenizer.vocab_size} max_len={tokenizer.max_len}")

    # ---- data ----
    cache = Path(args.latent_cache) if args.latent_cache else out_dir.parent / "latents_train.pt"
    latents, token_ids = precompute_latents(
        args.parquet_dir, "train", cache, tokenizer, device, limit=args.train_limit,
    )
    loader = DataLoader(
        TensorDataset(latents, token_ids),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=device == "cuda",
    )

    # ---- model ----
    from diffusers import DDPMScheduler
    from diffusers.training_utils import EMAModel

    unet = build_unet().to(device)
    encoder = ContentEncoder(tokenizer.vocab_size, tokenizer.max_len).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    params = list(unet.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    ema = EMAModel(unet.parameters(), decay=args.ema_decay)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    n_params = sum(p.numel() for p in params)
    print(f"[train] trainable params: {n_params/1e6:.1f}M — {len(loader)} steps/epoch")

    # ---- auto-resume ----
    start_epoch = 0
    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        unet.load_state_dict(state["unet"])
        encoder.load_state_dict(state["encoder"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        # checkpoint restores the old LR; honor the CLI value for resumed epochs
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        print(f"[train] Resumed from epoch {start_epoch} (lr={args.lr})")

    null_ids = torch.full((args.batch_size, tokenizer.max_len), PAD_ID, dtype=torch.long, device=device)
    null_ids[:, 0] = NULL_ID

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        running = 0.0
        for step, (lat, ids) in enumerate(loader):
            lat, ids = lat.to(device, non_blocking=True), ids.to(device, non_blocking=True)
            if torch.rand(()) < args.cfg_dropout:
                ids = null_ids[: ids.shape[0]]

            noise = torch.randn_like(lat)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=device)
            noisy = scheduler.add_noise(lat, noise, timesteps)

            with torch.autocast(device_type=device if device != "mps" else "cpu",
                                enabled=device == "cuda"):
                cond = encoder(ids)
                pred = unet(noisy, timesteps, encoder_hidden_states=cond).sample
                loss = F.mse_loss(pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.step(unet.parameters())
            running += loss.item()

        avg = running / len(loader)
        print(f"[train] epoch {epoch + 1}/{args.epochs}  loss {avg:.4f}", flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save({
                "epoch": epoch, "unet": unet.state_dict(), "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(), "ema": ema.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"mode": args.mode, "vocab_size": tokenizer.vocab_size,
                           "max_len": tokenizer.max_len, "cond_dim": COND_DIM},
            }, ckpt_path)
            print(f"[train] checkpoint @ epoch {epoch + 1} -> {ckpt_path}")

        if (epoch + 1) % args.preview_every == 0 or epoch + 1 == args.epochs:
            try:
                from paper2.ldm.sample_ldm import generate_grid
                preview = out_dir / f"preview_epoch{epoch + 1:04d}.png"
                generate_grid(unet, encoder, ema, tokenizer, device, preview)
                print(f"[train] preview -> {preview}")
            except Exception as exc:
                print(f"[train] preview failed (non-fatal): {exc}")

    print("[train] DONE")


if __name__ == "__main__":
    main()
