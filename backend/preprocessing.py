"""
DevGen Framework — Advanced Preprocessing Pipeline
Handles: Binarization, Deskewing, Denoising, Normalization
"""

import cv2
import numpy as np
from PIL import Image
import io


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to OpenCV BGR image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB Image."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive binarization for handwritten documents.
    Uses Gaussian adaptive thresholding to handle uneven lighting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=10
    )
    return binary


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct document skew using Hough Line Transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:  # Only consider near-horizontal lines
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def denoise(img: np.ndarray) -> np.ndarray:
    """Non-local means denoising for handwritten documents."""
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def normalize_for_model(img: np.ndarray, target_height: int = 384,
                         target_width: int = 384) -> np.ndarray:
    """Resize image to target dimensions while maintaining aspect ratio.
    Pads with white if needed."""
    h, w = img.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas and center the image
    if len(img.shape) == 3:
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    else:
        canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255

    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def crop_to_foreground(img: np.ndarray, padding_ratio: float = 0.18) -> np.ndarray:
    """Crop around visible handwriting while keeping a little context."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    h, w = gray.shape[:2]
    min_area = max(12, int(h * w * 0.0001))
    boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= min_area]
    if not boxes:
        return img

    x1 = min(x for x, _, _, _ in boxes)
    y1 = min(y for _, y, _, _ in boxes)
    x2 = max(x + bw for x, _, bw, _ in boxes)
    y2 = max(y + bh for _, y, _, bh in boxes)

    pad_x = max(8, int((x2 - x1) * padding_ratio))
    pad_y = max(8, int((y2 - y1) * padding_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img[y1:y2, x1:x2]


def preprocess_for_ocr(image_bytes: bytes) -> Image.Image:
    """Prepare a word image for TrOCR.
    Uses adaptive preprocessing based on the raw aspect ratio:
    - Long words (AR > 2.2) like 'khanekura' are cropped and padded to square.
    - Medium words (1.55 < AR <= 2.2) like 'nepal' and 'manish' are kept raw.
    - Short words/blocks (AR <= 1.55) are just cropped to foreground.
    """
    img = bytes_to_cv2(image_bytes)
    h, w = img.shape[:2]
    aspect_ratio = w / float(h)
    
    if aspect_ratio <= 1.55:
        img = crop_to_foreground(img)
    elif aspect_ratio > 2.2:
        img = crop_to_foreground(img)
        img = normalize_for_model(img)
        
    return cv2_to_pil(img)


def full_preprocess(image_bytes: bytes) -> Image.Image:
    """Complete preprocessing pipeline for document images.
    Steps: Denoise → Deskew → Binarize → Normalize → PIL"""
    img = bytes_to_cv2(image_bytes)
    img = denoise(img)
    img = deskew(img)
    # Keep color for ViT input (TrOCR expects RGB images)
    img = normalize_for_model(img)
    return cv2_to_pil(img)
