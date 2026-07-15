"""
Sample from a trained Devanagari handwriting LDM (paper2/ldm/train_ldm.py).

Two entry points:
- generate_grid(): quick 8-word preview grid during training.
- CLI: generate one image per word from a vocab file into the extracted
  format (images/ + labels.csv) that content_fidelity.py / compute_fid.py
  and train_trocr.py --synthetic-dir consume.

Usage:
    python -m paper2.ldm.sample_ldm \
        --checkpoint paper2/runs/ldm_akshara/checkpoint.pt \
        --vocab-file paper2/experiments/vocab/oov_unseen_conjunct.txt \
        --out-dir data_synth/v2_akshara_oov_unseen --guidance 2.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper2.ldm.tokenizer import NULL_ID, PAD_ID, DevanagariTokenizer  # noqa: E402

VAE_SCALE = 0.18215
LATENT_SHAPE = (4, 8, 32)  # C, H, W for 64x256 images

PREVIEW_WORDS = ["नमस्ते", "विद्यालय", "क्षत्रिय", "हिंदी", "लक्ष्मी", "भारत", "अनुसन्धान", "ज्ञान"]


@torch.no_grad()
def sample_batch(unet, encoder, tokenizer, words, device, vae=None,
                 steps: int = 50, guidance: float = 2.5, seed: int = 42):
    from diffusers import AutoencoderKL, DDIMScheduler

    if vae is None:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scheduler.set_timesteps(steps, device=device)

    ids = torch.tensor([tokenizer.encode(w) for w in words], dtype=torch.long, device=device)
    null_ids = torch.full_like(ids, PAD_ID)
    null_ids[:, 0] = NULL_ID
    cond = encoder(ids)
    uncond = encoder(null_ids)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn((len(words), *LATENT_SHAPE), generator=generator).to(device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        inp = scheduler.scale_model_input(latents, t)
        eps_c = unet(inp, t, encoder_hidden_states=cond).sample
        eps_u = unet(inp, t, encoder_hidden_states=uncond).sample
        eps = eps_u + guidance * (eps_c - eps_u)
        latents = scheduler.step(eps, t, latents).prev_sample

    images = vae.decode(latents / VAE_SCALE).sample
    images = ((images.clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
    return [Image.fromarray(arr) for arr in images], vae


def generate_grid(unet, encoder, ema, tokenizer, device, out_path: Path,
                  words=PREVIEW_WORDS, steps: int = 30) -> None:
    """Training-time preview using EMA weights (restored afterwards)."""
    unet.eval()
    ema.store(unet.parameters())
    ema.copy_to(unet.parameters())
    try:
        # guidance 1.5 — the 2.5 default oversaturates and made previews unreadable
        images, _ = sample_batch(unet, encoder, tokenizer, words, device, steps=steps, guidance=1.5)
        rows = len(images)
        grid = Image.new("RGB", (256, 64 * rows), "white")
        for i, img in enumerate(images):
            grid.paste(img, (0, i * 64))
        grid.save(out_path)
    finally:
        ema.restore(unet.parameters())
        unet.train()


def load_model(checkpoint: str, device: str, use_ema: bool = True):
    from diffusers.training_utils import EMAModel
    from paper2.ldm.train_ldm import ContentEncoder, build_unet

    state = torch.load(checkpoint, map_location=device)
    cfg = state["config"]
    tokenizer = DevanagariTokenizer.load(Path(checkpoint).parent / "tokenizer.json")

    unet = build_unet().to(device)
    unet.load_state_dict(state["unet"])
    encoder = ContentEncoder(cfg["vocab_size"], cfg["max_len"]).to(device)
    encoder.load_state_dict(state["encoder"])
    if use_ema:
        ema = EMAModel(unet.parameters())
        ema.load_state_dict(state["ema"])
        ema.copy_to(unet.parameters())
    unet.eval()
    encoder.eval()
    return unet, encoder, tokenizer, state["epoch"] + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate word images from a trained LDM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1, help="Images per word (distinct seeds).")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    unet, encoder, tokenizer, epoch = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    print(f"[sample] {tokenizer.mode} model @ epoch {epoch} on {device}")

    words = [w.strip() for w in Path(args.vocab_file).read_text(encoding="utf-8").splitlines() if w.strip()]
    tasks = [(w, r) for r in range(args.repeat) for w in words]

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    vae = None
    for start in range(0, len(tasks), args.batch_size):
        chunk = tasks[start : start + args.batch_size]
        seed = args.seed + start
        images, vae = sample_batch(
            unet, encoder, tokenizer, [w for w, _ in chunk], device, vae=vae,
            steps=args.steps, guidance=args.guidance, seed=seed,
        )
        for (word, rep), img in zip(chunk, images):
            filename = f"gen_{len(rows):06d}.png"
            img.save(images_dir / filename)
            rows.append({"filename": filename, "text": word, "seed": str(seed), "repeat": str(rep)})
        print(f"[sample] {min(start + args.batch_size, len(tasks))}/{len(tasks)}")

    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "text", "seed", "repeat"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sample] Wrote {len(rows)} images -> {out_dir}")


if __name__ == "__main__":
    main()
