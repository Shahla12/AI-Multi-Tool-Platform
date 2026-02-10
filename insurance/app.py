from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib  # <-- for loading saved model

# FASTAPI APP
app = FastAPI(title="Insurance Charges Prediction")

templates = Jinja2Templates(directory="templates")

# LOAD SAVED MODEL
model = joblib.load(r"C:\ML Tasks\insurance(regg)\insurance_model.pkl")
# Make sure this path points to your saved .pkl file

# HOME PAGE
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# PREDICTION FORM SUBMIT
@app.post("/predict")
async def predict(request: Request):
    form = await request.form()

    # Prepare input data exactly as your model expects
    input_df = pd.DataFrame([{
        'age': int(form["age"]),
        'bmi': float(form["bmi"]),
        'children': int(form["children"]),
        'sex': form["sex"],
        'smoker': form["smoker"],
        'region': form["region"]
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": round(float(prediction), 2)
    })
