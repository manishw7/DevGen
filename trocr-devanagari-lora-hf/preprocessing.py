import cv2
import numpy as np
from PIL import Image
import io

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def crop_to_foreground(img: np.ndarray, padding_ratio: float = 0.18) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    h, w = gray.shape[:2]
    min_area = max(12, int(h * w * 0.0001))
    boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= min_area]
    if not boxes: return img
    x1, y1, x2, y2 = min(x for x,_,_,_ in boxes), min(y for _,y,_,_ in boxes), max(x+bw for x,_,bw,_ in boxes), max(y+bh for _,y,_,bh in boxes)
    pad_x, pad_y = max(8, int((x2 - x1) * padding_ratio)), max(8, int((y2 - y1) * padding_ratio))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    return img[y1:y2, x1:x2]

def normalize_for_model(img: np.ndarray, target_height: int = 384, target_width: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    y_offset, x_offset = (target_height - new_h) // 2, (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def preprocess_for_ocr(image_bytes: bytes) -> Image.Image:
    img = bytes_to_cv2(image_bytes)
    if img is None: return None
    h, w = img.shape[:2]
    aspect_ratio = w / float(h)
    if aspect_ratio <= 1.55:
        img = crop_to_foreground(img)
    elif aspect_ratio > 2.2:
        img = crop_to_foreground(img)
        img = normalize_for_model(img)
    return cv2_to_pil(img)
