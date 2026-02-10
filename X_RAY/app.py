import os
import uuid
import numpy as np
import cv2
import tensorflow as tf

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------------
# APP SETUP
# --------------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------
# LOAD MODEL
# --------------------------------
MODEL_PATH = r"X_RAY\xray_model.keras"
model = load_model(MODEL_PATH)

# --------------------------------
# GRAD-CAM FUNCTION (UNCHANGED LOGIC)
# --------------------------------
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
    cam = cam / np.max(cam)
    return cam

# --------------------------------
# HOME PAGE
# --------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --------------------------------
# PREDICT + GRAD-CAM
# --------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict_xray(request: Request, file: UploadFile = File(...)):

    ext = file.filename.split(".")[-1]
    uid = f"{uuid.uuid4()}.{ext}"
    image_path = os.path.join(UPLOAD_DIR, uid)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # ---------- Prediction ----------
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    # confidence = float(prediction)

    # ---------- Grad-CAM ----------
    heatmap = gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    result_name = f"gradcam_{uid}"
    result_path = os.path.join(UPLOAD_DIR, result_name)
    cv2.imwrite(result_path, superimposed)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "uploaded": f"/static/uploads/{uid}",
        "result": f"/static/uploads/{result_name}",
        "label": label,
        # "confidence": f"{confidence:.4f}"
    })
