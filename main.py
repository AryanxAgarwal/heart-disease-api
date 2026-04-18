from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

app = FastAPI()

# Load your Random Forest model
model = joblib.load('heart_disease_model.pkl')

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
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <div class="container mt-5 text-center">
            <div class="alert {alert_class} p-5 shadow">
                <h1>{result_text}</h1>
                <hr>
                <a href="http://127.0.0.1:5500/frontend/index.html" class="btn btn-outline-dark">Analyze Another Patient</a>
            </div>
        </div>
    """)