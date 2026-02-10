import os
import uuid
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# ------------------ CONFIG ------------------
os.makedirs("static", exist_ok=True)

MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {device}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Cityscapes color palette
PALETTE = np.array([
    [128, 64,128], [244, 35,232], [70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170,30], [220,220,0], [107,142,35], [152,251,152],
    [70,130,180], [220,20,60], [255,0,0], [0,0,142], [0,0,70],
    [0,60,100], [0,80,100], [0,0,230], [119,11,32]
])

# ------------------ FASTAPI ------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------ FRONTEND ------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------ SEGMENT API ------------------
@app.post("/segment", response_class=HTMLResponse)
async def segment_image(request: Request, file: UploadFile = File(...)):

    uid = str(uuid.uuid4())
    input_path = f"static/{uid}_input.jpg"
    output_path = f"static/{uid}_segmented.png"

    # Save input image
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Load & upscale image
    image = Image.open(input_path).convert("RGB")
    high_res = image.resize((2048, 1024))

    # Prepare model input
    inputs = processor(images=high_res, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=high_res.size[::-1], mode="bilinear", align_corners=False
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # Color segmentation
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    for label_id, color in enumerate(PALETTE):
        color_seg[pred_seg == label_id] = color

    # Smooth segmentation
    color_seg = cv2.bilateralFilter(color_seg, 9, 75, 75)

    # Save output
    Image.fromarray(color_seg).save(output_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "original": input_path,
        "segmented": output_path
    })

# ------------------ RUN ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
