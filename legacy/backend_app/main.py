"""
DevGen Framework — FastAPI Backend
Neuro-Generative Suite for Devanagari Handwritten Text Recognition

Endpoints:
  GET  /                          → Health check
  GET  /api/v1/dataset/info       → Dataset metadata
  GET  /api/v1/dataset/sample     → Single sample
  GET  /api/v1/dataset/random     → Random samples
  GET  /api/v1/dataset/browse     → Paginated samples
  POST /api/v1/recognize          → OCR recognition
  POST /api/v1/recognize/full     → OCR + NER in one shot
  POST /api/v1/preprocess         → Preprocess an image
  POST /api/v1/evaluate           → CER evaluation
  POST /api/v1/ner                → NER extraction from plain text
  POST /api/v1/generate           → Generate synthetic handwriting
  GET  /api/v1/generate/info      → GAN model info
"""

from fastapi import FastAPI, UploadFile, File, Form, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
import io
from PIL import Image
import os
import traceback

from backend.preprocessing import full_preprocess, preprocess_for_ocr
from backend.dataset_loader import get_dataset_info, get_sample, get_random_samples, get_paginated_samples
from backend.ner_extractor import extract_entities, summarize_entities
from backend.trocr_engine import CERCalculator, TrOCREngine, inspect_model_artifacts

# Lazy-load heavy ML models
_trocr_engine = None


def get_trocr_engine():
    """Lazy-load the TrOCR engine to avoid long startup times."""
    global _trocr_engine
    if _trocr_engine is None:
        _trocr_engine = TrOCREngine()
    return _trocr_engine


app = FastAPI(
    title="DevGen Framework API",
    description="Neuro-Generative Suite for Devanagari Handwritten Text Recognition",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    model_info = inspect_model_artifacts()
    return {
        "status": "ok",
        "service": "DevGen Framework",
        "version": "0.1.0",
        "modules": ["preprocessing", "trocr_engine", "dataset_loader", "ner_extractor"],
        "model_loaded": _trocr_engine is not None,
        "model": {
            "base_model_name": model_info["base_model_name"],
            "adapter_path": model_info["adapter_path"],
            "adapter_source": model_info["adapter_source"],
            "device": model_info["device"],
            "using_adapter": model_info["using_adapter"],
        },
    }


@app.get("/api/v1/model/info")
def model_info():
    """Return model configuration and checkpoint discovery status."""
    try:
        if _trocr_engine is not None:
            return {"status": "ok", "data": _trocr_engine.info()}

        discovered = inspect_model_artifacts()
        discovered["loaded"] = False
        discovered["default_max_length"] = int(os.getenv("TROCR_MAX_LENGTH", "128"))
        discovered["num_beams"] = 4
        return {"status": "ok", "data": discovered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dataset Endpoints ──────────────────────────────────────────────────────

@app.get("/api/v1/dataset/info")
def dataset_info():
    """Get dataset metadata (splits, sizes, columns)."""
    try:
        info = get_dataset_info()
        return {"status": "ok", "data": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dataset/sample")
def dataset_sample(
    split: str = Query("train", description="Dataset split: train, validation, test"),
    index: int = Query(0, description="Sample index"),
):
    """Get a single dataset sample by index."""
    try:
        sample = get_sample(split, index)
        return {"status": "ok", "data": sample}
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dataset/random")
def dataset_random(
    split: str = Query("train", description="Dataset split"),
    count: int = Query(12, description="Number of random samples"),
):
    """Get random samples from the dataset."""
    try:
        samples = get_random_samples(split, count)
        return {"status": "ok", "data": samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dataset/browse")
def dataset_browse(
    split: str = Query("train", description="Dataset split"),
    page: int = Query(0, description="Page number (0-indexed)"),
    page_size: int = Query(20, description="Samples per page"),
):
    """Paginated dataset browsing."""
    try:
        result = get_paginated_samples(split, page, page_size)
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Preprocessing Endpoint ─────────────────────────────────────────────────

@app.post("/api/v1/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """Preprocess a document image (denoise, deskew, normalize).
    Returns the preprocessed image as base64."""
    try:
        contents = await file.read()
        processed_pil = full_preprocess(contents)

        buffer = io.BytesIO()
        processed_pil.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "ok",
            "filename": file.filename,
            "preprocessed_image_base64": b64,
            "size": {"width": processed_pil.width, "height": processed_pil.height},
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Recognition Endpoint ───────────────────────────────────────────────────

@app.post("/api/v1/recognize")
async def recognize_document(
    request: Request,
    file: UploadFile = File(...)
):
    """Full OCR pipeline: Preprocess → TrOCR Recognition.
    Returns recognized text with per-token confidence scores."""
    try:
        form = await request.form()
        force_model = form.get("force_model")
        
        if force_model == "":
            force_model = None
            
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Recognition (Internal preprocessing now handled by the engine)
        engine = get_trocr_engine()
        print(f"[API] recognize_document: force_model='{force_model}' (form keys: {list(form.keys())})")
        result = engine.recognize(image, force_model=force_model)
        
        # We can still provide the preprocessed image for visual feedback
        processed_image = engine._preprocess_image(image)
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")
        result["preprocessed_image_base64"] = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")
        
        result["filename"] = file.filename
        result["status"] = "ok"
        
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation Endpoint ────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    prediction: str
    reference: str


@app.post("/api/v1/evaluate")
def evaluate_cer(req: EvalRequest):
    """Calculate Character Error Rate between prediction and reference."""
    try:
        result = CERCalculator.calculate(req.prediction, req.reference)
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── NER Endpoint ───────────────────────────────────────────────────────────

class NERRequest(BaseModel):
    text: str


@app.post("/api/v1/ner")
def extract_ner(req: NERRequest):
    """Extract structured entities (dates, IDs, districts) from recognized text."""
    try:
        entities = extract_entities(req.text)
        summary = summarize_entities(entities)
        return {"status": "ok", "data": {"entities": entities, "summary": summary}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recognize/full")
async def recognize_and_extract(
    request: Request,
    file: UploadFile = File(...)
):
    """Full pipeline: Preprocess → TrOCR Recognition → NER Extraction.
    Returns recognized text, confidence scores, AND structured entities."""
    try:
        form = await request.form()
        force_model = form.get("force_model")
        
        if force_model == "":
            force_model = None
            
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Step 1: Recognition (Internal preprocessing now handled by the engine)
        engine = get_trocr_engine()
        print(f"[API] recognize_and_extract: force_model='{force_model}' (form keys: {list(form.keys())})")
        result = engine.recognize(image, force_model=force_model)

        # Step 2: NER Extraction
        entities = extract_entities(result["text"])
        result["entities"] = entities
        result["entities_summary"] = summarize_entities(entities)

        # Include preprocessed image
        processed_image = engine._preprocess_image(image)
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")
        result["preprocessed_image_base64"] = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")
        
        result["filename"] = file.filename
        result["status"] = "ok"

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


_gan_generator = None


def _load_gan():
    """Lazy-load the GAN generator model."""
    global _gan_generator
    if _gan_generator is not None:
        return _gan_generator

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    from backend.trocr_engine import get_best_torch_device
    device = get_best_torch_device()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = ["hybrid_step_80000.pt", "hybrid_step_35000.pt", "upgraded_step_30000.pt", "wgan_generator_final.pth", "generator_v2.pth"]
    model_path = None
    for name in candidates:
        path = os.path.join(project_root, name)
        if os.path.exists(path):
            model_path = path
            break
    if model_path is None:
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
        is_upgraded = True
    else:
        state_dict = checkpoint
        is_upgraded = False

    if is_upgraded:
        LATENT_DIM = 128
        EMBED_DIM = 128
        STYLE_DIM = 256
        vocab_size = len(checkpoint.get("char_to_idx", {})) + 1

        class LightweightBottleneckAttention(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                neck = max(in_channels // 8, 1)
                self.query = nn.Conv2d(in_channels, neck, 1)
                self.key   = nn.Conv2d(in_channels, neck, 1)
                self.value = nn.Conv2d(in_channels, in_channels, 1)
                self.gamma = nn.Parameter(torch.zeros(1))
            
            def forward(self, x):
                B, C, H, W = x.size()
                q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
                k = self.key(x).view(B, -1, H * W)
                v = self.value(x).view(B, -1, H * W)
                scale = q.size(-1) ** -0.5
                attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
                out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
                return x + self.gamma * out

        class StyleModulatedConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.demodulate = demodulate
                self.padding = kernel_size // 2
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
                nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                self.style_proj = nn.Linear(style_dim, in_channels)
                self.style_proj.bias.data.fill_(1.0)
            
            def forward(self, x, style):
                B, C, H, W = x.size()
                s = self.style_proj(style).reshape(B, 1, C, 1, 1)
                w = self.weight.unsqueeze(0) * s
                if self.demodulate:
                    demod = torch.rsqrt(w.pow(2).sum(dim=[2, 3, 4]) + 1e-8)
                    w = w * demod.reshape(B, self.out_channels, 1, 1, 1)
                x = x.reshape(1, B * C, H, W)
                w = w.reshape(B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
                return F.conv2d(x, w, padding=self.padding, groups=B).reshape(B, self.out_channels, H, W)

        class NoiseInjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
            
            def forward(self, x):
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
                return x + self.weight * noise

        class SpatialTextEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.gru = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(embed_dim, 512)
            
            def forward(self, text):
                embedded = self.embedding(text)
                outputs, hidden = self.gru(embedded)
                spatial_feats = self.fc(outputs).permute(0, 2, 1).unsqueeze(2)  # [B, 512, 1, seq_len]
                global_feat = torch.cat([hidden[0], hidden[1]], dim=-1)         # [B, embed_dim]
                return spatial_feats, global_feat

        class StyleMappingNetwork(nn.Module):
            def __init__(self, latent_dim, style_dim):
                super().__init__()
                layers = []
                for i in range(6):
                    in_dim = latent_dim if i == 0 else style_dim
                    layers.extend([nn.Linear(in_dim, style_dim), nn.LeakyReLU(0.2, True)])
                self.net = nn.Sequential(*layers)
            
            def forward(self, z):
                z = z / (torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True)) + 1e-8)
                return self.net(z)

        class ModulatedUpsampleBlock(nn.Module):
            def __init__(self, in_chan, out_chan, style_dim):
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.conv = StyleModulatedConv2d(in_chan, out_chan, 3, style_dim)
                self.noise = NoiseInjection()
                self.lrelu = nn.LeakyReLU(0.2, True)
            
            def forward(self, x, style):
                x = self.up(x)
                x = self.conv(x, style)
                x = self.noise(x)
                return self.lrelu(x)

        class ModulatedRefineBlock(nn.Module):
            def __init__(self, chan, style_dim):
                super().__init__()
                self.conv1 = StyleModulatedConv2d(chan, chan, 3, style_dim)
                self.noise1 = NoiseInjection()
                self.lrelu = nn.LeakyReLU(0.2, True)
                self.conv2 = StyleModulatedConv2d(chan, chan, 3, style_dim)
                self.noise2 = NoiseInjection()
            
            def forward(self, x, style):
                residual = x
                x = self.noise1(self.conv1(x, style))
                x = self.lrelu(x)
                x = self.noise2(self.conv2(x, style))
                return residual + x

        class AdvancedStyleGANAttentionGenerator(nn.Module):
            def __init__(self, style_dim):
                super().__init__()
                self.expand_height = nn.ConvTranspose2d(512, 512, kernel_size=(4, 1), stride=(4, 1))
                self.bottleneck_attention = LightweightBottleneckAttention(512)
                self.up1 = ModulatedUpsampleBlock(512, 256, style_dim)
                self.up2 = ModulatedUpsampleBlock(256, 128, style_dim)
                self.up3 = ModulatedUpsampleBlock(128, 64, style_dim)
                self.up4 = ModulatedUpsampleBlock(64, 32, style_dim)
                self.refine = ModulatedRefineBlock(32, style_dim)
                self.out = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Tanh())
                
                self.to_rgb1 = nn.Conv2d(256, 1, 1)
                self.to_rgb2 = nn.Conv2d(128, 1, 1)
                self.to_rgb3 = nn.Conv2d(64, 1, 1)
                self.to_rgb4 = nn.Conv2d(32, 1, 1)
            
            def forward(self, spatial_feats, style):
                x = self.expand_height(spatial_feats)
                x = self.bottleneck_attention(x)
                x = self.up1(x, style)
                rgb = self.to_rgb1(x)
                x = self.up2(x, style)
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.to_rgb2(x)
                x = self.up3(x, style)
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.to_rgb3(x)
                x = self.up4(x, style)
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.to_rgb4(x)
                x = self.refine(x, style)
                out = self.out(x)
                return torch.tanh(out + rgb)

        gen = AdvancedStyleGANAttentionGenerator(STYLE_DIM)
        # Load EMA generator if available (ema_state_dict), fallback to generator_state_dict
        if "ema_state_dict" in checkpoint:
            gen.load_state_dict(checkpoint["ema_state_dict"])
            print(f"[GAN] Loaded EMA generator state dict from {os.path.basename(model_path)}")
        else:
            gen.load_state_dict(state_dict)
            print(f"[GAN] Loaded standard generator state dict from {os.path.basename(model_path)}")
        
        gen.to(device)
        gen.eval()

        encoder = SpatialTextEncoder(vocab_size, EMBED_DIM)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.to(device)
        encoder.eval()

        mapping_net = StyleMappingNetwork(LATENT_DIM, STYLE_DIM)
        mapping_net.load_state_dict(checkpoint["mapping_state_dict"])
        mapping_net.to(device)
        mapping_net.eval()

        _gan_generator = {
            "model": gen,
            "encoder": encoder,
            "mapping_net": mapping_net,
            "char_to_idx": checkpoint.get("char_to_idx"),
            "vocab": checkpoint.get("vocab"),
            "device": device,
            "latent_dim": LATENT_DIM,
            "is_upgraded": True
        }
        print(f"[GAN] StyleGAN-Attention Generator loaded successfully from {os.path.basename(model_path)} (256x64) on {device}")
        return _gan_generator

    else:
        # Dynamically determine resolution from weight dimensions to support both 64x128 and 128x256
        LATENT_DIM = 128
        l1_weight_shape = state_dict["l1.weight"].shape[0]
        init_h = int(math.sqrt(l1_weight_shape / 1024))
        h_dim = init_h * 16
        w_dim = h_dim * 2

        class SelfAttention(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
                self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
                self.value = nn.Conv2d(in_channels, in_channels, 1)
                self.gamma = nn.Parameter(torch.zeros(1))
                self.softmax = nn.Softmax(dim=-1)
                
            def forward(self, x):
                b, c, h, w = x.size()
                q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
                k = self.key(x).view(b, -1, h * w)
                v = self.value(x).view(b, -1, h * w)
                attn = self.softmax(torch.bmm(q, k))
                out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
                return self.gamma * out + x

        class WGANGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.init_h, self.init_w = h_dim // 16, w_dim // 16
                self.l1 = nn.Linear(LATENT_DIM, 512 * self.init_h * self.init_w)
                
                self.block1 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True), nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False))
                self.block2 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True), nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False))
                self.attn1 = SelfAttention(128)
                self.block3 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True), nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False))
                self.block4 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False))
                self.out = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Tanh())

            def forward(self, z):
                x = self.l1(z).view(-1, 512, self.init_h, self.init_w)
                x = self.block1(x)
                x = self.block2(x)
                x = self.attn1(x)
                x = self.block3(x)
                x = self.block4(x)
                return self.out(x)

        gen = WGANGenerator()
        gen.load_state_dict(state_dict)
        gen.to(device)
        gen.eval()

        _gan_generator = {
            "model": gen,
            "device": device,
            "latent_dim": LATENT_DIM,
            "is_upgraded": False
        }
        print(f"[GAN] Unconditional WGAN Generator loaded from {os.path.basename(model_path)} ({w_dim}x{h_dim}) on {device}")
    return _gan_generator


class GenerateRequest(BaseModel):
    count: int = 4
    text: Optional[str] = None


@app.post("/api/v1/generate")
def generate_handwriting(req: GenerateRequest):
    """Generate synthetic Devanagari handwriting images using the trained GAN."""
    import torch
    from fastapi import HTTPException
    import numpy as np
    import io
    import base64

    gan = _load_gan()
    if gan is None:
        raise HTTPException(status_code=503, detail="GAN model not found. Place generator_ema.pth or generator.pth in project root.")

    count = min(req.count, 16)  # Cap at 16
    model, device, z_dim = gan["model"], gan["device"], gan["latent_dim"]

    with torch.inference_mode():
        if gan.get("is_upgraded"):
            # Prepare conditioning text
            from backend.dataset_loader import get_random_samples
            if req.text:
                words = [req.text] * count
            else:
                try:
                    samples = get_random_samples("train", count)
                    words = [s["text"] for s in samples]
                except Exception:
                    fallback_words = ["नेपाल", "काठमाडौं", "देवनागरी", "भारत", "विकास", "शिक्षा", "स्वास्थ्य", "नेपाली", "भाषा", "ज्ञान"]
                    words = [fallback_words[i % len(fallback_words)] for i in range(count)]

            char_to_idx = gan["char_to_idx"]
            max_len = 16
            padded_tokens_list = []
            for word in words:
                tokens = [char_to_idx[c] for c in word if c in char_to_idx][:max_len]
                t_len = len(tokens)
                if t_len == 0:
                    tokens = [1]
                    t_len = 1
                # Center aligned padding to match training exact config
                left_pad = (max_len - t_len) // 2
                right_pad = max_len - t_len - left_pad
                padded = [0] * left_pad + tokens + [0] * right_pad
                padded_tokens_list.append(padded)

            tokens_tensor = torch.tensor(padded_tokens_list, dtype=torch.long, device=device)
            spatial_feats, _ = gan["encoder"](tokens_tensor)

            z = torch.randn(count, z_dim, device=device)
            style = gan["mapping_net"](z)
            images = model(spatial_feats, style)
        else:
            z = torch.randn(count, z_dim, device=device)
            images = model(z)

    # Convert each image to base64
    from PIL import Image as PILImage

    results = []
    for i in range(count):
        img_tensor = images[i].cpu()
        # Denormalize from [-1,1] to [0,255]
        img_np = ((img_tensor.squeeze(0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        results.append({"image_base64": b64, "index": i})

    return {"status": "ok", "count": count, "images": results}


@app.get("/api/v1/generate/info")
def generate_info():
    """Return GAN model availability and metadata."""
    import torch
    import math
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = ["hybrid_step_80000.pt", "hybrid_step_35000.pt", "upgraded_step_30000.pt", "generator_ema.pth", "generator.pth", "generator_v2.pth"]
    model_path = None
    for name in candidates:
        path = os.path.join(project_root, name)
        if os.path.exists(path):
            model_path = path
            break
    available = model_path is not None
    size_mb = round(os.path.getsize(model_path) / 1e6, 1) if available else 0
    
    resolution = "64x64"
    latent_dim = 128
    architecture = "ResNet GAN v3 (DiffAugment + Hinge + EMA)"

    if available:
        basename = os.path.basename(model_path)
        if basename in ["hybrid_step_80000.pt", "hybrid_step_35000.pt", "upgraded_step_30000.pt"]:
            resolution = "64x256"
            latent_dim = 128
            architecture = "Advanced StyleGAN-Attention Generator (Spatial GRU + Bottleneck Self-Attention)"
        else:
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                l1_weight_shape = checkpoint["l1.weight"].shape[0]
                init_h = int(math.sqrt(l1_weight_shape / 1024))
                h_dim = init_h * 16
                w_dim = h_dim * 2
                resolution = f"{h_dim}x{w_dim}"
            except Exception:
                resolution = "128x256"

    return {
        "status": "ok",
        "data": {
            "available": available,
            "model_path": model_path,
            "model_file": os.path.basename(model_path) if model_path else None,
            "model_size_mb": size_mb,
            "architecture": architecture,
            "resolution": resolution,
            "latent_dim": latent_dim,
        }
    }


# ── LDM Generation Endpoint ──────────────────────────────────────────────────

@app.get("/api/v1/generate/ldm/info")
def get_ldm_info():
    """Return active LDM ControlNet model info."""
    try:
        from backend.ldm_engine import get_active_checkpoint_info
        info = get_active_checkpoint_info()
        return {"status": "ok", "data": info}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class LDMGenerateRequest(BaseModel):
    text: str
    font_path: str = "font.ttf"
    conditioning_scale: float = 1.0

@app.post("/api/v1/generate/ldm")
def generate_handwriting_ldm_endpoint(req: LDMGenerateRequest):
    """Generate synthetic handwriting using LDM ControlNet + UNet LoRA."""
    try:
        from backend.ldm_engine import generate_handwriting_ldm
        
        output_image, control_image = generate_handwriting_ldm(
            text=req.text,
            font_path=req.font_path,
            conditioning_scale=req.conditioning_scale
        )
        
        # Convert generated output to base64
        buf_out = io.BytesIO()
        output_image.save(buf_out, format="PNG")
        b64_out = base64.b64encode(buf_out.getvalue()).decode("utf-8")
        
        # Convert control map to base64
        buf_ctrl = io.BytesIO()
        control_image.save(buf_ctrl, format="PNG")
        b64_ctrl = base64.b64encode(buf_ctrl.getvalue()).decode("utf-8")
        
        return {
            "status": "ok", 
            "text": req.text,
            "generated_image_base64": b64_out,
            "control_image_base64": b64_ctrl
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
