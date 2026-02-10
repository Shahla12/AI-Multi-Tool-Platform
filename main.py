import warnings
warnings.filterwarnings("ignore")
import torchvision
torchvision.disable_beta_transforms_warning()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import uuid
import pickle
import joblib
import numpy as np
import pandas as pd
import cv2
import torch
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForImageSegmentation
import torchvision.transforms as T
import shutil
import tempfile
from dotenv import load_dotenv

# Helper for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel filesystem is read-only. Use /tmp for uploads.
IS_VERCEL = "VERCEL" in os.environ
UPLOAD_DIR = "/tmp/uploads" if IS_VERCEL else os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Mount static and templates
# Note: On Vercel, static files are handled by the platform, but /tmp/uploads won't be servable via StaticFiles easily.
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==========================================
# 1. LAZY MODEL LOADING
# ==========================================

MODEL_CACHE = {}

def get_model(name):
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]
    
    print(f"Loading model: {name}...")
    
    # Check if model file exists before loading
    def check_and_load(path, load_fn, is_binary=True):
        if not os.path.exists(path):
            if IS_VERCEL:
                raise FileNotFoundError(f"Model file not found at {path}. Note: Large models are ignored by Git. You must host them externally for Vercel.")
            raise FileNotFoundError(f"Model file not found at {path}. Please ensure you have the models downloaded.")
        
        if is_binary:
            with open(path, 'rb') as f:
                return load_fn(f)
        return load_fn(path)

    if name == "house_pricing":
        model_path = os.path.join(BASE_DIR, "House Pricing", "house_price_model.pkl")
        model = check_and_load(model_path, joblib.load, is_binary=False)
    elif name == "sms_spam":
        model_path = os.path.join(BASE_DIR, "sms_spam_detection", "spam_detect_model.pkl")
        model = check_and_load(model_path, pickle.load)
    elif name == "sms_vectorizer":
        model_path = os.path.join(BASE_DIR, "sms_spam_detection", "tfidf_vectorizer.pkl")
        model = check_and_load(model_path, pickle.load)
    elif name == "xray":
        model_path = os.path.join(BASE_DIR, "X_RAY", "xray_model.keras")
        model = check_and_load(model_path, load_model, is_binary=False)
    elif name == "headline":
        model_path = os.path.join(BASE_DIR, "headline", "news_headlines_bilstm_v3.keras")
        model = check_and_load(model_path, load_model, is_binary=False)
    elif name == "headline_tokenizer":
        model_path = os.path.join(BASE_DIR, "headline", "tokenizer.pkl")
        model = check_and_load(model_path, pickle.load)
    elif name == "insurance":
        model_path = os.path.join(BASE_DIR, "insurance", "insurance_model.pkl")
        model = check_and_load(model_path, joblib.load, is_binary=False)
    elif name == "student_score":
        model_path = os.path.join(BASE_DIR, "student_score", "student_score_predictor.pkl")
        model = check_and_load(model_path, joblib.load, is_binary=False)
    elif name == "segmentation_processor":
        model = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    elif name == "segmentation_model":
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        model.eval()
    elif name == "mask_model":
        model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
        model.eval()
    elif name == "mask_transform":
        model = T.Compose([
            T.ToTensor(),
            T.Resize((1024, 1024)),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    
    MODEL_CACHE[name] = model
    return model

# Cityscapes color palette
PALETTE = np.array([
    [128, 64,128], [244, 35,232], [70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170,30], [220,220,0], [107,142,35], [152,251,152],
    [70,130,180], [220,20,60], [255,0,0], [0,0,142], [0,0,70],
    [0,60,100], [0,80,100], [0,0,230], [119,11,32]
])

# ==========================================
# ROUTES
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "dashboard"})

@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    return FileResponse(os.path.join(UPLOAD_DIR, filename))

# --- House Pricing ---
@app.get("/house-pricing", response_class=HTMLResponse)
async def house_pricing_get(request: Request):
    return templates.TemplateResponse("house_pricing.html", {"request": request, "active_page": "house_pricing"})

@app.post("/house-pricing", response_class=HTMLResponse)
async def house_pricing_post(
    request: Request,
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    mainroad: str = Form(...),
    guestroom: str = Form(...),
    basement: str = Form(...),
    airconditioning: str = Form(...),
    parking: int = Form(...),
    furnishingstatus: str = Form(...)
):
    input_data = pd.DataFrame([{
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
        "airconditioning": airconditioning, "parking": parking,
        "furnishingstatus": furnishingstatus
    }])
    model = get_model("house_pricing")
    prediction = model.predict(input_data)[0]
    return templates.TemplateResponse("house_pricing.html", {
        "request": request,
        "prediction": f"{round(prediction, 2):,}",
        "active_page": "house_pricing"
    })

# --- SMS Spam ---
@app.get("/sms-spam", response_class=HTMLResponse)
async def sms_spam_get(request: Request):
    return templates.TemplateResponse("sms_spam.html", {"request": request, "active_page": "sms_spam"})

@app.post("/sms-spam", response_class=HTMLResponse)
async def sms_spam_post(request: Request, message: str = Form(...)):
    vectorizer = get_model("sms_vectorizer")
    model = get_model("sms_spam")
    # Preprocess: convert to lowercase
    message_clean = message.lower()
    message_tfidf = vectorizer.transform([message_clean])
    prediction = model.predict(message_tfidf)[0]
    # Get prediction probability for confidence
    prediction_proba = model.predict_proba(message_tfidf)[0]
    confidence = round(max(prediction_proba) * 100, 2)
    result = "SPAM" if prediction == 1 else "HAM"
    return templates.TemplateResponse("sms_spam.html", {
        "request": request,
        "prediction": result,
        "confidence": confidence,
        "message": message,
        "active_page": "sms_spam"
    })

# --- Insurance ---
@app.get("/insurance", response_class=HTMLResponse)
async def insurance_get(request: Request):
    return templates.TemplateResponse("insurance.html", {"request": request, "active_page": "insurance"})

@app.post("/insurance", response_class=HTMLResponse)
async def insurance_post(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...)
):
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi,
        'children': children, 'smoker': smoker, 'region': region
    }])
    model = get_model("insurance")
    prediction = model.predict(input_df)[0]
    return templates.TemplateResponse("insurance.html", {
        "request": request,
        "prediction": f"{round(prediction, 2):,}",
        "active_page": "insurance"
    })

# --- Student Score ---
@app.get("/student-score", response_class=HTMLResponse)
async def student_score_get(request: Request):
    return templates.TemplateResponse("student_score.html", {"request": request, "active_page": "student_score"})

@app.post("/student-score", response_class=HTMLResponse)
async def student_score_post(request: Request, hours: float = Form(...)):
    model = get_model("student_score")
    prediction = model.predict([[hours]])[0]
    return templates.TemplateResponse("student_score.html", {
        "request": request,
        "prediction": round(prediction, 2),
        "hours": hours,
        "active_page": "student_score"
    })

# --- Headline Classify ---
@app.get("/headline", response_class=HTMLResponse)
async def headline_get(request: Request):
    return templates.TemplateResponse("headline.html", {"request": request, "active_page": "headline"})

@app.post("/headline", response_class=HTMLResponse)
async def headline_post(request: Request, headline: str = Form(...)):
    tokenizer = get_model("headline_tokenizer")
    model = get_model("headline")
    sequences = tokenizer.texts_to_sequences([headline])
    padded = pad_sequences(sequences, maxlen=50, padding="post", truncating="post")
    preds = model.predict(padded)
    categories = ["Business", "Sci/Tech", "Sports", "World"]
    category = categories[int(np.argmax(preds, axis=1)[0])]
    return templates.TemplateResponse("headline.html", {
        "request": request,
        "headline": headline,
        "prediction": category,
        "active_page": "headline"
    })

# --- X-RAY ---
def gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-10)
    return cam

@app.get("/xray", response_class=HTMLResponse)
async def xray_get(request: Request):
    return templates.TemplateResponse("xray.html", {"request": request, "active_page": "xray"})

@app.post("/xray", response_class=HTMLResponse)
async def xray_post(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    input_filename = f"{uid}_input.{ext}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Prediction
    model = get_model("xray")
    img = tf_image.load_img(input_path, target_size=(224, 224))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

    # Grad-CAM
    heatmap = gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(input_path)
    original = cv2.resize(original, (224, 224))
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    result_filename = f"{uid}_gradcam.png"
    result_path = os.path.join(UPLOAD_DIR, result_filename)
    cv2.imwrite(result_path, superimposed)

    return templates.TemplateResponse("xray.html", {
        "request": request,
        "active_page": "xray",
        "uploaded": f"/uploads/{input_filename}",
        "result": f"/uploads/{result_filename}",
        "label": label
    })

# --- Cartoonify ---
def cartoon_simple_black_outline_flat_colors_from_bgr(frame_bgr, line_thickness=2, n_colors=24):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
    h, w = img.shape[:2]
    img_flat = (img * 255).reshape(-1, 3).astype(np.float32)
    k = max(2, min(n_colors, img_flat.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, labels, centers = cv2.kmeans(img_flat, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        flat_colors = centers[labels.flatten()].reshape((h, w, 3)).astype(np.float32) / 255.0
    except:
        flat_colors = (np.floor(img * 255 / (256 // k)) * (256 // k)) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 80, 160)
    kt = max(1, int(line_thickness))
    kernel = np.ones((kt, kt), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_mask = (edges > 0).astype(np.float32)
    edges_mask = np.stack([edges_mask] * 3, axis=-1)
    cartoon_result = flat_colors * (1 - edges_mask)
    cartoon_bgr = (np.clip(cartoon_result, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cartoon_bgr, cv2.COLOR_RGB2BGR)

@app.get("/cartoon", response_class=HTMLResponse)
async def cartoon_get(request: Request):
    return templates.TemplateResponse("cartoon.html", {"request": request, "active_page": "cartoon"})

@app.post("/cartoon", response_class=HTMLResponse)
async def cartoon_post(
    request: Request,
    file: UploadFile = File(...),
    n_colors: int = Form(12),
    line_thickness: int = Form(2)
):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_input.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_cartoon.png")
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
    
    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return templates.TemplateResponse("cartoon.html", {"request": request, "error": "Invalid image", "active_page": "cartoon"})
    
    cartoon_bgr = cartoon_simple_black_outline_flat_colors_from_bgr(img_bgr, line_thickness, n_colors)
    cv2.imwrite(output_path, cartoon_bgr)
    
    return templates.TemplateResponse("cartoon.html", {
        "request": request,
        "active_page": "cartoon",
        "uploaded": f"/uploads/{uid}_input.png",
        "result": f"/uploads/{uid}_cartoon.png"
    })

# --- Segmentation ---
@app.get("/segmentation", response_class=HTMLResponse)
async def segmentation_get(request: Request):
    return templates.TemplateResponse("segmentation.html", {"request": request, "active_page": "segmentation"})

@app.post("/segmentation", response_class=HTMLResponse)
async def segmentation_post(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_seg_in.jpg")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_seg_out.png")
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    image = Image.open(input_path).convert("RGB")
    high_res = image.resize((2048, 1024))
    processor = get_model("segmentation_processor")
    model = get_model("segmentation_model")
    inputs = processor(images=high_res, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=high_res.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    for label_id, color in enumerate(PALETTE):
        color_seg[pred_seg == label_id] = color
    
    color_seg = cv2.bilateralFilter(color_seg, 9, 75, 75)
    Image.fromarray(color_seg).save(output_path)
    
    return templates.TemplateResponse("segmentation.html", {
        "request": request,
        "active_page": "segmentation",
        "uploaded": f"/uploads/{uid}_seg_in.jpg",
        "result": f"/uploads/{uid}_seg_out.png"
    })

# --- Masking / BG Removal ---
@app.get("/masking", response_class=HTMLResponse)
async def masking_get(request: Request):
    return templates.TemplateResponse("masking.html", {"request": request, "active_page": "masking"})

@app.post("/masking", response_class=HTMLResponse)
async def masking_post(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_mask_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_mask_out.png")
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    img = Image.open(input_path).convert("RGB")
    transform = get_model("mask_transform")
    model = get_model("mask_model")
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        result = model(input_tensor)
    
    mask = result[0][0].cpu().numpy()
    mask = np.squeeze(mask)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_resized = mask_img.resize(img.size, Image.LANCZOS)
    
    mask_np = np.array(mask_resized) / 255.0
    mask_np = np.expand_dims(mask_np, axis=2)
    img_np = np.array(img).astype(np.float32) / 255.0
    bg_removed = img_np * mask_np
    
    Image.fromarray((bg_removed * 255).astype(np.uint8)).save(output_path)
    
    return templates.TemplateResponse("masking.html", {
        "request": request,
        "active_page": "masking",
        "uploaded": f"/uploads/{uid}_mask_in.png",
        "result": f"/uploads/{uid}_mask_out.png"
    })



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
