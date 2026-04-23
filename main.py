from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import joblib
import pandas as pd
import os
from typing import Optional

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Brain Tumor Prediction API")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and scaler globally
model = None
scaler = None
global_top_features = []

try:
    if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logger.info("Model and Scaler loaded successfully.")
        
        # Extract feature importances dynamically globally
        feature_names = ['Age', 'Tumor Size', 'Survival Rate', 'Tumor Growth Rate', 'Stage', 'Tumor Density', 'Edema Size']
        importances = model.feature_importances_
        sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        # Formatted top 3 features for UI consumption
        global_top_features = [{"name": f[0], "importance": f"{f[1]*100:.1f}%"} for f in sorted_features[:3]]
    else:
        logger.warning("Model and Scaler not found. Run train_model.py first.")
except Exception as e:
    logger.error(f"Error loading model/scaler: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Render the landing info page by default."""
    logger.info("Serving Application Landing Page.")
    return templates.TemplateResponse("info.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request) -> HTMLResponse:
    """Render the prediction form for users to input tumor parameters."""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history_get(request: Request) -> HTMLResponse:
    """Render the history logging dashboard cleanly accessing local bindings seamlessly."""
    return templates.TemplateResponse("history.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    age: int = Form(..., description="Patient age, must be 0-100"),
    tumor_size: float = Form(..., description="Tumor size in mm"),
    survival_rate: float = Form(..., description="Estimated survival rate percentage"),
    tumor_growth_rate: float = Form(..., description="Growth rate on a 1-10 scale"),
    stage: int = Form(..., description="Tumor stage, must be 1-4"),
    tumor_density: float = Form(..., description="Density of the tumor mass"),
    edema_size: float = Form(..., description="Size of surrounding edema in mm")
) -> HTMLResponse:
    """Handle prediction submissions dynamically applying preprocessing and the XGBoost model."""
    error: Optional[str] = None
    prediction: Optional[str] = None
    confidence: float = 0.0
    
    input_data = {
        "Age": age,
        "Tumor Size": tumor_size,
        "Survival Rate": survival_rate,
        "Tumor Growth Rate": tumor_growth_rate,
        "Stage": stage,
        "Tumor Density": tumor_density,
        "Edema Size": edema_size
    }
    
    # Input validation
    if not (0 <= age <= 100):
        error = "Age must be between 0 and 100."
    elif not (1 <= stage <= 4):
        error = "Stage must be between 1 and 4."
    elif model is None or scaler is None:
        error = "Internal System Configuration Missing: Please contact administrator or verify 'models/' configuration."
        logger.error("Attempted prediction but model/scaler pipeline maps are None.")

    # Model prediction workflow
    if error is None:
        try:
            logger.info(f"Processing prediction for Age: {age}, Stage: {stage}")
            # Create DataFrame mapping exactly to the 7 features
            df_input = pd.DataFrame(
                [[age, tumor_size, survival_rate, tumor_growth_rate, stage, tumor_density, edema_size]], 
                columns=['Age', 'Tumor_Size', 'Survival_Rate', 'Tumor_Growth_Rate', 'Stage', 'Tumor_Density', 'Edema_Size']
            )
            
            # Application of standard scaler
            X_scaled = scaler.transform(df_input)
            
            # Prediction and Confidence logic integration 
            pred = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            
            # Mapping max probability out as explicit percentage
            prediction = "Malignant" if pred == 1 else "Benign"
            confidence = float(max(probs) * 100)
            
            logger.info(f"Generated Prediction Result -> {prediction} ({confidence:.2f}%)")
            
        except Exception as e:
            logger.exception("An unexpected error occurred parsing metrics bounds during XGBoost prediction")
            error = "Prediction calculation failed due to an unexpected system error."

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction,
        "confidence": confidence,
        "input_data": input_data,
        "top_features": global_top_features,
        "error": error
    })

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    
    print("\n🚀 Server starting...")
    print(f"👉 Open in browser: http://localhost:{port}\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=port)
