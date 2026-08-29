from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import uvicorn
import os

# -------------------------------------------------------------------
# Load ML Models
# -------------------------------------------------------------------
MODEL_PATH = "model-kidney-desease.pkl"
SCALER_PATH = "scaller-kidney-desease.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model files: {str(e)}")

# -------------------------------------------------------------------
# FastAPI App Initialization & CORS Setup
# -------------------------------------------------------------------
app = FastAPI(
    title="Chronic Kidney Disease Prediction API",
    description="ML Inference service for React & Laravel integration",
    version="1.0.0"
)

# Enable CORS for React and Laravel origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production (e.g., ["http://localhost:3000", "http://localhost:8000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Input Request Schema Validation
# -------------------------------------------------------------------
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, example=48)
    bp: float = Field(..., ge=40, le=200, example=80)
    sg: float = Field(..., ge=1.000, le=1.060, example=1.020)
    al: float = Field(..., ge=0.0, le=5.0, example=1.0)
    hemo: float = Field(..., ge=5.0, le=20.0, example=15.4)
    sc: float = Field(..., ge=0.5, le=10.0, example=1.2)
    htn: str = Field(..., example="yes")
    dm: str = Field(..., example="no")
    cad: str = Field(..., example="no")
    appet: str = Field(..., example="good")
    pc: str = Field(..., example="normal")

class PredictionResponse(BaseModel):
    prediction: int
    has_disease: bool
    message: str

# -------------------------------------------------------------------
# Category Mappings (Replaces single-fit LabelEncoder)
# -------------------------------------------------------------------
BINARY_MAP = {"no": 0, "yes": 1}
APPET_MAP = {"good": 0, "poor": 1}
PC_MAP = {"normal": 0, "abnormal": 1}

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Kidney Disease Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_disease(data: PatientData):
    try:
        # 1. Normalize and Map Categorical Variables
        htn_enc = BINARY_MAP.get(data.htn.lower())
        dm_enc = BINARY_MAP.get(data.dm.lower())
        cad_enc = BINARY_MAP.get(data.cad.lower())
        appet_enc = APPET_MAP.get(data.appet.lower())
        pc_enc = PC_MAP.get(data.pc.lower())

        if None in [htn_enc, dm_enc, cad_enc, appet_enc, pc_enc]:
            raise HTTPException(status_code=400, detail="Invalid categorical value provided.")

        # 2. Build DataFrame matching original training feature order
        input_data = {
            'age': [data.age],
            'bp': [data.bp],
            'sg': [data.sg],
            'al': [data.al],
            'hemo': [data.hemo],
            'sc': [data.sc],
            'htn': [htn_enc],
            'dm': [dm_enc],
            'cad': [cad_enc],
            'appet': [appet_enc],
            'pc': [pc_enc]
        }
        df = pd.DataFrame(input_data)

        # 3. Scale numeric features using the pre-fitted scaler
        numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 4. Predict
        prediction = int(model.predict(df)[0])
        has_disease = (prediction == 0)

        return PredictionResponse(
            prediction=prediction,
            has_disease=has_disease,
            message="The patient is likely to have Chronic Kidney Disease." if has_disease else "The patient is NOT likely to have Chronic Kidney Disease."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)