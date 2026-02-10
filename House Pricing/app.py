from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("House Pricing\house_price_model.pkl")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
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

    # Convert input data into DataFrame
    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "airconditioning": airconditioning,
        "parking": parking,
        "furnishingstatus": furnishingstatus
    }])

    prediction = model.predict(input_data)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction, 2)
        }
    )
