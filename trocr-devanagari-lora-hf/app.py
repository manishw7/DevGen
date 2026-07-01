import os
import io
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from peft import PeftModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from cnn_model import CharacterClassifier
from preprocessing import preprocess_for_ocr

# --- CONFIGURATION ---
BASE_MODEL_ID = "paudelanil/trocr-devanagari-2"
ADAPTER_ID = "manishw10/devgen-trocr-devanagari-lora"
CNN_MODEL_PATH = "devanagari-cnn-classifier.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ENGINE INITIALIZATION ---
print("System: Initializing Stable Premium Engine (3.50.2)...")
processor = TrOCRProcessor.from_pretrained(BASE_MODEL_ID)
base_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_ID)
base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
base_model.config.pad_token_id = processor.tokenizer.pad_token_id
base_model.config.eos_token_id = processor.tokenizer.sep_token_id
base_model.config.vocab_size = base_model.config.decoder.vocab_size

peft_model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
try:
    model = peft_model.merge_and_unload()
except Exception:
    model = peft_model
model.to(device); model.eval()
cnn_engine = CharacterClassifier(model_path=CNN_MODEL_PATH, device=device)

# --- ORIGINAL ROUTING LOGIC ---
def _flood_fill(binary, visited, start_y, start_x, h, w):
    stack = [(start_y, start_x)]
    size = 0
    while stack:
        y, x = stack.pop()
        if y<0 or y>=h or x<0 or x>=w or visited[y,x] or not binary[y,x]: continue
        visited[y,x] = True; size += 1
        stack.extend([(y+1,x),(y-1,x),(y,x+1),(y,x-1)])
    return size

def count_blobs(binary):
    h, w = binary.shape; visited = np.zeros_like(binary, dtype=bool); count = 0
    for y in range(h):
        for x in range(w):
            if binary[y,x] and not visited[y,x]:
                size = _flood_fill(binary, visited, y, x, h, w)
                if size >= max(binary.size * 0.001, 10): count += 1
    return count

def original_classify_input(image):
    gray = image.convert("L"); arr = np.array(gray)
    threshold = min(arr.mean() * 0.75, 200)
    binary = (arr < threshold).astype(np.uint8)
    rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)
    if not rows.any() or not cols.any(): return "character", 1.0, 1
    y0, x0 = np.where(rows)[0][0], np.where(cols)[0][0]
    y1, x1 = np.where(rows)[0][-1], np.where(cols)[0][-1]
    w, h = x1-x0+1, y1-y0+1
    ar, bc = w/h, count_blobs(binary)
    is_char = True
    if ar > 2.5: is_char = False
    elif ar > 1.8 and bc >= 3: is_char = False
    elif bc >= 4: is_char = False
    elif ar < 1.3 and bc <= 2: is_char = True
    elif bc == 1 and ar < 1.5: is_char = True
    elif ar < 1.75 and bc <= 2: is_char = True
    elif ar > 1.6: is_char = False
    return ("character" if is_char else "word"), ar, bc

def get_confidence_html(confidence):
    color = "#10b981" if confidence > 0.9 else "#f59e0b" if confidence > 0.7 else "#ef4444"
    return f"""<div style="display: flex; flex-direction: column; align-items: center; background: rgba(0,0,0,0.2); border-radius: 20px; padding: 15px;">
        <svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8" />
            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8" stroke-dasharray="282.7" stroke-dashoffset="{282.7 * (1 - confidence)}" stroke-linecap="round" />
            <text x="50" y="55" font-family="Arial" font-size="20" font-weight="bold" fill="{color}" text-anchor="middle">{int(confidence * 100)}%</text>
        </svg>
    </div>"""

# --- PREDICT ---
def predict(image, manual_mode):
    if image is None: return None, None, "Upload image.", "", ""
    buf = io.BytesIO(); image.save(buf, format="PNG")
    pre_pil = preprocess_for_ocr(buf.getvalue())
    if manual_mode == "Automatic":
        mode, ar, bc = original_classify_input(pre_pil)
        status = f"System: {mode.upper()} (AR: {ar:.2f}, Blobs: {bc})"
    else:
        mode = manual_mode.lower(); status = f"Manual Mode: {mode.upper()}"
    try:
        if mode == "character" and cnn_engine.available:
            res = cnn_engine.predict(pre_pil)
            return pre_pil, res["text"], status, "CNN Classifier", get_confidence_html(res["confidence"])
        else:
            pixel_values = processor(pre_pil, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                out = model.generate(pixel_values, num_beams=4, max_length=128, early_stopping=True, return_dict_in_generate=True, output_scores=True, decoder_start_token_id=model.config.decoder_start_token_id)
            scores = torch.exp(model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0])
            txt = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
            return pre_pil, txt, status, "TrOCR + LoRA", get_confidence_html(float(scores.mean().item()))
    except Exception as e:
        return pre_pil, f"Error: {str(e)}", "Failed", "None", ""

# --- PREMIUM CSS (Gradio 3.x Optimized) ---
CSS = """
.gradio-container { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important; color: white !important; }
.premium-card { background: rgba(30, 41, 59, 0.7) !important; border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
.result-box textarea { font-size: 2.5rem !important; font-weight: bold !important; color: #818cf8 !important; text-align: center !important; background: transparent !important; border: none !important; }
h1 { color: #818cf8 !important; font-size: 2.5rem !important; }
"""

with gr.Blocks(css=CSS) as demo:
    with gr.Column(elem_classes="premium-card"):
        gr.Markdown("# 🕉️ DevGen OCR")
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Input")
                mode_ctrl = gr.Radio(["Automatic", "Word", "Character"], value="Automatic", label="Mode")
                sub_btn = gr.Button("Recognize", variant="primary")
            with gr.Column(scale=1):
                conf_html = gr.HTML()
                text_out = gr.Textbox(label="Result", elem_classes="result-box", interactive=False)
                status_md = gr.Markdown("Ready.")
                engine_txt = gr.Textbox(label="Model", interactive=False)
        with gr.Column():
            gr.Markdown("### 🛠️ Visual Debug: What the Model Sees")
            img_proc = gr.Image(type="pil", label="Preprocessed", interactive=False)

    sub_btn.click(predict, [img_in, mode_ctrl], [img_proc, text_out, status_md, engine_txt, conf_html])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
