import os
import uuid
from typing import Optional

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import cv2
from PIL import Image
import io

# Optional TensorFlow model loading (only if you want to use your .keras model)
MODEL = None
MODEL_PATH = "cartoon_model.keras"
try:
    if os.path.exists(MODEL_PATH):
        import tensorflow as tf
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded model:", MODEL_PATH)
except Exception as e:
    print("Model load skipped / failed:", e)
    MODEL = None

# Create folders
BASE_STATIC = "static"
UPLOAD_DIR = os.path.join(BASE_STATIC, "uploads")
OUTPUT_DIR = os.path.join(BASE_STATIC, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def unique_filename(ext="png"):
    return f"{uuid.uuid4().hex}.{ext}"


def read_image_from_bytesfile(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # BGR uint8 or None


def save_bgr_image(img_bgr: np.ndarray, path: str):
    cv2.imwrite(path, img_bgr)


def rgb_norm_to_bgr_uint8(img_rgb_norm: np.ndarray) -> np.ndarray:
    rgb = (np.clip(img_rgb_norm, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_norm(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def apply_model_if_available(frame_bgr: np.ndarray):
    if MODEL is None:
        return frame_bgr
    try:
        rgb = bgr_to_rgb_norm(frame_bgr)
        inp = np.expand_dims(rgb, 0).astype(np.float32)
        pred = MODEL.predict(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = np.squeeze(pred)
        out_bgr = rgb_norm_to_bgr_uint8(pred)
        return out_bgr
    except Exception as e:
        print("Model inference failed, returning original:", e)
        return frame_bgr


def cartoon_simple_black_outline_flat_colors_from_bgr(frame_bgr: np.ndarray,
                                                     line_thickness: int = 2,
                                                     n_colors: int = 24) -> np.ndarray:
    """
    Accept BGR uint8 frame and return BGR uint8 cartoonified image.
    """
    # Convert BGR to RGB and scale to [0,1]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
    h, w = img.shape[:2]

    # Step 1: Reduce colors using k-means
    img_flat = (img * 255).reshape(-1, 3).astype(np.float32)
    # ensure k <= number of pixels
    k = max(2, min(n_colors, img_flat.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, labels, centers = cv2.kmeans(img_flat, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        flat_colors = centers[labels.flatten()].reshape((h, w, 3)).astype(np.float32) / 255.0
    except Exception as e:
        # fallback: use quantization by uniform binning if kmeans fails
        print("kmeans failed, fallback quantization:", e)
        flat_colors = (np.floor(img * 255 / (256 // k)) * (256 // k)) / 255.0

    # Step 2: Detect edges
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 80, 160)

    # Step 3: Dilate edges for thicker outline
    kt = max(1, int(line_thickness))
    kernel = np.ones((kt, kt), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edges_mask = (edges > 0).astype(np.float32)
    edges_mask = np.stack([edges_mask] * 3, axis=-1)

    # Step 4: Combine black edges with flat colors
    cartoon_result = flat_colors * (1 - edges_mask)
    cartoon_result += edges_mask * 0  # black edges

    # Convert back to BGR 0-255
    cartoon_bgr = (np.clip(cartoon_result, 0, 1) * 255).astype(np.uint8)
    cartoon_bgr = cv2.cvtColor(cartoon_bgr, cv2.COLOR_RGB2BGR)
    return cartoon_bgr


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/cartoon", response_class=HTMLResponse)
async def cartoonize(request: Request,
                     file: UploadFile = File(...),
                     n_colors: Optional[int] = Form(12),
                     line_thickness: Optional[int] = Form(2),
                     use_model: Optional[bool] = Form(False)):
    # read bytes
    contents = await file.read()
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    upload_name = unique_filename(ext)
    upload_path = os.path.join(UPLOAD_DIR, upload_name)
    with open(upload_path, "wb") as f:
        f.write(contents)

    img_bgr = read_image_from_bytesfile(contents)
    if img_bgr is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Could not read uploaded image. Please upload a valid PNG/JPG image."
        })

    processed = img_bgr
    if use_model and MODEL is not None:
        processed = apply_model_if_available(processed)

    try:
        cartoon_bgr = cartoon_simple_black_outline_flat_colors_from_bgr(
            processed,
            line_thickness=max(1, int(line_thickness)),
            n_colors=max(2, int(n_colors))
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Cartoonify failed: {e}"
        })

    out_name = unique_filename("png")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    save_bgr_image(cartoon_bgr, out_path)

    upload_url = f"/static/uploads/{upload_name}"
    out_url = f"/static/outputs/{out_name}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "uploaded": upload_url,
        "result": out_url,
        "n_colors": n_colors,
        "line_thickness": line_thickness,
        "use_model": use_model
    }) 



