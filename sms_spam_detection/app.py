from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



with open('sms_spam_detection\spam_detect_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('sms_spam_detection\\tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer =pickle.load(f)

# Initialize FastAPI
app =FastAPI()

# Setup templates folder
templates = Jinja2Templates(directory="templates")


# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    # Transform the input message using TF-IDF
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]

        # Convert prediction to readable label
    result = "SPAM" if prediction == 1 else "HAM"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})