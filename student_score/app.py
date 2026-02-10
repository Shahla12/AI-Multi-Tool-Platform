from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Load trained model
model = joblib.load("student_score/student_score_predictor.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, hours: float = Form(...)):

    
    if hours <= 0 or hours > 10:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                
            }
        )

    prediction = model.predict(np.array([[hours]]))[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction, 2),
            "hours": hours
        }
    )
