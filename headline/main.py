from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import uvicorn
import os

# ---------------------------
# Config
# ---------------------------
MAX_LEN = 50  # must match training

# Order must match training label encoding
CATEGORIES = ["Business", "Sci/Tech", "Sports", "World"]

# ---------------------------
# Load model and tokenizer
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "news_headlines_bilstm_v3.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

print("Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")

# ---------------------------
# FastAPI app and templates
# ---------------------------
app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "headline": "",
            "prediction": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_category(request: Request, headline: str = Form(...)):
    # 1) Text → sequences using SAME tokenizer as training
    sequences = tokenizer.texts_to_sequences([headline])

    # 2) Pad to same maxlen and settings as training
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    # 3) Predict
    preds = model.predict(padded)
    pred_index = int(np.argmax(preds, axis=1)[0])
    category = CATEGORIES[pred_index]

    # Debug (optional – remove in production)
    print("Headline:", headline)
    print("Sequences:", sequences)
    print("Padded[0][:10]:", padded[0][:10])
    print("Preds:", preds[0])
    print("Argmax index:", pred_index)
    print("Chosen label:", category)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "headline": headline,
            "prediction": category,
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
