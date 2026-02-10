import os
import uuid
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torchvision.transforms as T
from transformers import AutoModelForImageSegmentation

# ------------------ SETUP ------------------

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------ LOAD MODEL ONCE ------------------

model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-1.4",
    trust_remote_code=True
).eval()

transform = T.Compose([
    T.ToTensor(),
    T.Resize((1024, 1024)),
    T.Normalize([0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])
])

# ------------------ ROUTES ------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/remove-bg", response_class=HTMLResponse)
async def remove_bg(request: Request, file: UploadFile = File(...)):

    try:
        # ----- SAVE UPLOADED IMAGE -----
        image_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{image_id}_input.png")
        output_path = os.path.join(UPLOAD_DIR, f"{image_id}_output.png")

        with open(input_path, "wb") as f:
            f.write(await file.read())

        # ----- LOAD IMAGE -----
        img = Image.open(input_path).convert("RGB")

        input_tensor = transform(img).unsqueeze(0)

        # ----- PREDICT MASK -----
        with torch.no_grad():
            result = model(input_tensor)

        mask = result[0][0].cpu().numpy()
        mask = np.squeeze(mask)
        mask = (mask * 255).astype(np.uint8)

        mask_img = Image.fromarray(mask, mode="L")
        mask_resized = mask_img.resize(img.size, Image.LANCZOS)

        # ----- REMOVE BACKGROUND -----
        mask_np = np.array(mask_resized) / 255.0
        mask_np = np.expand_dims(mask_np, axis=2)

        img_np = np.array(img).astype(np.float32) / 255.0
        bg_removed = img_np * mask_np

        bg_removed_img = Image.fromarray((bg_removed * 255).astype(np.uint8))
        bg_removed_img.save(output_path)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "uploaded": f"/static/results/{image_id}_input.png",
            "result": f"/static/results/{image_id}_output.png"
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })

