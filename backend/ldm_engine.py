import os
import torch
import numpy as np
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from PIL import Image, ImageDraw, ImageFont

_ldm_pipeline = None
_current_checkpoint = None

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Checkpoint search paths (newest first)
CHECKPOINT_SEARCH_DIRS = [
    os.path.join(PROJECT_ROOT, "ldm", "controlnet_devanagari_v4"),
    os.path.join(PROJECT_ROOT, "ldm", "controlnet_devanagari_v3"),
]
DEFAULT_FONT_PATH = os.path.join(BACKEND_DIR, "font.ttf")
IMG_SIZE = 256  # Must match training


def render_text_conditioning(text, font_path=DEFAULT_FONT_PATH, img_size=IMG_SIZE):
    """
    Render Devanagari text as white-on-black conditioning image.
    This MUST be identical to the training pipeline's render_text_conditioning().
    """
    base_size = 80
    try:
        font = ImageFont.truetype(font_path, size=base_size)
    except Exception:
        font = ImageFont.load_default()

    # Render on large temp canvas
    temp = Image.new("L", (1024, 512), color=0)
    draw = ImageDraw.Draw(temp)
    draw.text((50, 50), text, fill=255, font=font)

    # Auto-crop to tight bounding box
    np_temp = np.array(temp)
    ys, xs = np.where(np_temp > 0)
    if len(ys) == 0:
        return Image.new("RGB", (img_size, img_size), color="black")

    cropped = temp.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    cw, ch = cropped.size

    # Scale to fit inside 80% of canvas, preserving aspect ratio
    max_dim = int(img_size * 0.80)
    scale = min(max_dim / cw, max_dim / ch)
    new_w = max(8, int(cw * scale))
    new_h = max(8, int(ch * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on square canvas
    canvas = Image.new("RGB", (img_size, img_size), color="black")
    x = (img_size - new_w) // 2
    y = (img_size - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _find_controlnet_checkpoint():
    """
    Find the best available ControlNet checkpoint directory.
    Returns the path to the controlnet/ subfolder containing config.json + weights.
    """
    for base_dir in CHECKPOINT_SEARCH_DIRS:
        if not os.path.isdir(base_dir):
            continue
        # Find latest checkpoint-XXXX inside
        ckpts = [
            d for d in os.listdir(base_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))
        ]
        if not ckpts:
            continue
        latest = sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1]
        cn_dir = os.path.join(base_dir, latest, "controlnet")
        if os.path.exists(os.path.join(cn_dir, "config.json")):
            print(f"[LDM Engine] Found checkpoint: {os.path.join(base_dir, latest)}")
            return cn_dir

    raise FileNotFoundError(
        f"No ControlNet checkpoint found. Searched:\n"
        + "\n".join(f"  - {d}" for d in CHECKPOINT_SEARCH_DIRS)
    )


def get_active_checkpoint_info():
    """Returns information about the resolved active checkpoint."""
    try:
        cn_dir = _find_controlnet_checkpoint()
        checkpoint_dir = os.path.dirname(cn_dir)
        return {
            "available": True,
            "checkpoint_name": os.path.basename(checkpoint_dir),
            "checkpoint_path": checkpoint_dir,
            "resolution": f"{IMG_SIZE}x{IMG_SIZE}",
        }
    except Exception as e:
        return {
            "available": False,
            "checkpoint_name": "None",
            "checkpoint_path": None,
            "resolution": f"{IMG_SIZE}x{IMG_SIZE}",
            "error": str(e)
        }


def transliterate_devanagari(text):
    """Transliterates Devanagari Hindi/Nepali words phonetically into Latin characters."""
    char_map = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
        'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
        'ं': 'm', 'ः': 'h', 'ँ': 'n',
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'yn',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    }
    res = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in char_map:
            val = char_map[c]
            is_consonant = c in 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञ'
            if is_consonant:
                if i + 1 < n and text[i+1] == '्':
                    res.append(val)
                    i += 2
                    continue
                elif i + 1 < n and text[i+1] in 'ािीुूृेैोौ':
                    res.append(val + char_map[text[i+1]])
                    i += 2
                    continue
                else:
                    if i + 1 == n:
                        res.append(val)
                    else:
                        res.append(val + 'a')
                    i += 1
            else:
                res.append(val)
                i += 1
        else:
            res.append(c)
            i += 1
    return "".join(res)


def load_ldm_pipeline(checkpoint_path=None):
    """Load the ControlNet pipeline. Caches globally."""
    global _ldm_pipeline, _current_checkpoint

    cn_dir = checkpoint_path or _find_controlnet_checkpoint()

    # Return cached if same source
    if _ldm_pipeline is not None and _current_checkpoint == cn_dir:
        return _ldm_pipeline

    model_id = "runwayml/stable-diffusion-v1-5"
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[LDM Engine] Loading ControlNet from: {cn_dir}")
    controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # Use fast scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    _ldm_pipeline = pipe
    _current_checkpoint = cn_dir
    print(f"[LDM Engine] Pipeline ready on {device} (dtype={dtype})")
    return _ldm_pipeline


def generate_handwriting_ldm(text: str, checkpoint_path=None, font_path="font.ttf", conditioning_scale=1.0):
    """
    Generate synthetic Devanagari handwriting using the ControlNet pipeline.
    Returns (output_image, control_image) as PIL Images.
    """
    pipe = load_ldm_pipeline(checkpoint_path)

    # Resolve font path
    resolved_font = font_path if os.path.isabs(font_path) else os.path.join(BACKEND_DIR, os.path.basename(font_path))
    if not os.path.exists(resolved_font):
        resolved_font = DEFAULT_FONT_PATH

    # Generate conditioning image — identical to training
    control_image = render_text_conditioning(text, resolved_font)

    # Prompt — identical to training
    trans = transliterate_devanagari(text)
    prompt = f"handwritten Devanagari word '{trans}' on white paper, blue ink, high quality"
    negative_prompt = "blurry, low quality, digital font, typed, outline text"

    # Generator on correct device
    device = pipe.device
    gen_device = "cpu" if device.type == "mps" else device.type
    generator = torch.Generator(device=gen_device).manual_seed(42)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=float(conditioning_scale),
        generator=generator,
        width=IMG_SIZE,
        height=IMG_SIZE,
    ).images[0]

    return output, control_image