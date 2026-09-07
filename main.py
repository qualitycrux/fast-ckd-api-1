from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import uvicorn
import os
from pathlib import Path

# Build paths relative to this file's location
BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------
# Load ML Models
# -------------------------------------------------------------------
models = {}
scalers = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models['ckd_quick'] = pickle.load(open(BASE_DIR / 'model-kidney-desease.pkl', 'rb'))
    scalers['ckd_quick'] = pickle.load(open(BASE_DIR / 'scaller-kidney-desease.pkl', 'rb'))
    models['ckd_advance'] = pickle.load(open(BASE_DIR / 'model-kidney-disease.pkl', 'rb'))
    scalers['ckd_advance'] = pickle.load(open(BASE_DIR / 'scaller-kidney-disease.pkl', 'rb'))
    models['diabetic'] = pickle.load(open('model-diab-prediction.pkl', 'rb'))
    scalers['diabetic'] = pickle.load(open('scaler-diab-prediction.pkl', 'rb'))

    yield  # API active state

    print("Clearing models from memory...")
    models.clear()
    scalers.clear()

# -------------------------------------------------------------------
# FastAPI App Initialization & CORS Setup
# -------------------------------------------------------------------
app = FastAPI(
    title="Chronic Kidney Disease Prediction API",
    description="ML Inference service for React & Laravel integration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Input Request Schema Validation
# -------------------------------------------------------------------
class PatientDataQuick(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    hemo: float
    sc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pc: str

class PatientDataAdvance(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    hemo: float
    pcv: float
    rc: float
    bu: float
    bgr: float
    sc: float
    sod: float
    pot: float
    htn: str
    dm: str
    cad: str
    appet: str
    pc: str
    pe: str
    ane: str

class PatientDataDiabetic(BaseModel):
    pg: float
    gc: float
    bp: float
    sth: float
    ins: float
    bmi: float
    dpf: float
    age: float
    
class PredictionResponse(BaseModel):
    prediction: int
    has_disease: bool
    message: str

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Kidney Disease Prediction API"}

@app.post("/predict/quick", response_model=PredictionResponse)
def predict_disease_quick(data: PatientDataQuick):
    try:
        input_data = {
            'age': [data.age],
            'bp': [data.bp],
            'sg': [data.sg],
            'al': [data.al],
            'hemo': [data.hemo],
            'sc': [data.sc],
            'htn': [data.htn],
            'dm': [data.dm],
            'cad': [data.cad],
            'appet': [data.appet],
            'pc': [data.pc]
        }
        df = pd.DataFrame(input_data)

        le = LabelEncoder()
        df['htn'] = le.fit_transform(df['htn'])
        df['dm'] = le.fit_transform(df['dm'])
        df['cad'] = le.fit_transform(df['cad'])
        df['appet'] = le.fit_transform(df['appet'])
        df['pc'] = le.fit_transform(df['pc'])

        numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
        scaler = scalers["ckd_quick"]

        df[numeric_cols] = scaler.transform(df[numeric_cols])

        model = models['ckd_quick']
        prediction = int(model.predict(df)[0])
        has_disease = (prediction == 0)

        return PredictionResponse(
            prediction=prediction,
            has_disease=has_disease,
            message="The patient is likely to have Chronic Kidney Disease." if has_disease else "The patient is NOT likely to have Chronic Kidney Disease."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/advance", response_model=PredictionResponse)
def predict_disease_advance(data: PatientDataAdvance):
    try:
        input_data = {
            'age': [data.age],
            'bp': [data.bp],
            'sg': [data.sg],
            'al': [data.al],
            'hemo': [data.hemo],
            'pcv': [data.pcv],
            'rc': [data.rc],
            'bu': [data.bu],
            'bgr': [data.bgr],
            'sc': [data.sc],
            'sod': [data.sod],
            'pot': [data.pot],
            'htn': [data.htn],
            'dm': [data.dm],
            'cad': [data.cad],
            'appet': [data.appet],
            'pc': [data.pc],
            'pe': [data.pe],
            'ane': [data.ane],
        }
        df = pd.DataFrame(input_data)

        le = LabelEncoder()
        df['htn'] = le.fit_transform(df['htn'])
        df['dm'] = le.fit_transform(df['dm'])
        df['cad'] = le.fit_transform(df['cad'])
        df['appet'] = le.fit_transform(df['appet'])
        df['pc'] = le.fit_transform(df['pc'])
        df['pe'] = le.fit_transform(df['pe'])
        df['ane'] = le.fit_transform(df['ane'])

        numeric_cols = ['age','bp','sg','al','hemo','pcv','rc','bu','bgr','sc','sod','pot']
        scaler = scalers["ckd_advance"]

        df[numeric_cols] = scaler.transform(df[numeric_cols])

        model = models['ckd_advance']
        prediction = int(model.predict(df)[0])
        has_disease = (prediction == 0)

        return PredictionResponse(
            prediction=prediction,
            has_disease=has_disease,
            message="The patient is likely to have Chronic Kidney Disease." if has_disease else "The patient is NOT likely to have Chronic Kidney Disease."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# diabetic prediction
@app.post("/predict/diabetic", response_model=PredictionResponse)
def predict_disease_diabetic(data: PatientDataDiabetic):
    try:
        # 1. Build DataFrame matching original training feature order
        input_data = {
            'Pregnancies': [data.pg],
            'Glucose': [data.gc],
            'BloodPressure': [data.bp],
            'SkinThickness': [data.sth],
            'Insulin': [data.ins],
            'BMI': [data.bmi],
            'DiabetesPedigreeFunction': [data.dpf],
            'Age': [data.age]
        }

        # 2. Create the DataFrame
        df = pd.DataFrame(input_data)

        # 3. Scale features
        scaler = scalers["diabetic"]
        df = scaler.transform(df)

        # 4. Predict
        model = models['diabetic']
        raw_prediction = model.predict(df)  

        prediction_val = int(raw_prediction[0])
        has_disease = (prediction_val == 1)

        return PredictionResponse(
            prediction=prediction_val,  # Pass the native Python int
            has_disease=has_disease,
            message="The patient is likely to have Diabetic Disease." if has_disease else "The patient is NOT likely to have Diabetic Disease."
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Host changed to 0.0.0.0 to allow incoming external connections on Railway
    # Turned reload OFF for production stability
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
