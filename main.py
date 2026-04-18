from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# 1. ADD CORS MIDDLEWARE (Crucial for Netlify -> Render communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your Random Forest model
# Note: Ensure the .pkl file is in the same directory as this main.py on GitHub
try:
    model = joblib.load('heart_disease_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(
    age: int = Form(...), sex: int = Form(...), cp: int = Form(...),
    trestbps: int = Form(...), chol: int = Form(...), fbs: int = Form(...),
    restecg: int = Form(...), thalach: int = Form(...), exang: int = Form(...),
    oldpeak: float = Form(...), slope: int = Form(...), ca: int = Form(...),
    thal: int = Form(...)
):
    # Prepare data for prediction
    data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    df = pd.DataFrame(data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    prediction = model.predict(df)[0]
    
    # Bootstrap styled response page
    result_text = "Heart Disease Detected (High Risk)" if prediction == 1 else "No Heart Disease Detected (Low Risk)"
    alert_class = "alert-danger" if prediction == 1 else "alert-success"

    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <title>Diagnostic Result</title>
        </head>
        <body class="bg-light">
            <div class="container mt-5 text-center">
                <div class="alert {alert_class} p-5 shadow">
                    <h1 class="display-4 fw-bold">{result_text}</h1>
                    <hr>
                    <p class="lead">The analysis is based on the machine learning model's prediction.</p>
                    <a href="https://sparkling-sherbet-fa7764.netlify.app" class="btn btn-dark btn-lg mt-3">Analyze Another Patient</a>
                </div>
            </div>
        </body>
        </html>
    """)