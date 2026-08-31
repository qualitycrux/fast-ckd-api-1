from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import uvicorn
import os

# -------------------------------------------------------------------
# Load ML Models
# -------------------------------------------------------------------
models = {}
scalers={}
async def lifespan(app: FastAPI):
    models['ckd_quick'] = pickle.load(open('model-kidney-desease.pkl', 'rb'))
    scalers['ckd_quick'] = pickle.load(open('scaller-kidney-desease.pkl', 'rb'))
    models['ckd_advance'] = pickle.load(open('model-kidney-disease.pkl', 'rb'))
    scalers['ckd_advance'] = pickle.load(open('scaller-kidney-disease.pkl', 'rb'))

    yield  # The API runs while this line stays active

    # 2. Clean up or release memory during shutdown (Optional)
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

class PredictionResponse(BaseModel):
    prediction: int
    has_disease: bool
    message: str

# -------------------------------------------------------------------
# Category Mappings (Replaces single-fit LabelEncoder)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Kidney Disease Prediction API"}

@app.post("/predict/quick", response_model=PredictionResponse)
def predict_disease_quick(data: PatientDataQuick):
    try:


        # 1. Build DataFrame matching original training feature order
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

        # 2. Normalize and Categorical Variables
        le = LabelEncoder()
        df['htn'] = le.fit_transform(df['htn'])
        df['dm'] = le.fit_transform(df['dm'])
        df['cad'] = le.fit_transform(df['cad'])
        df['appet'] = le.fit_transform(df['appet'])
        df['pc'] = le.fit_transform(df['pc'])

        # 3. Scale numeric features using the pre-fitted scaler
        numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
        scaler=scalers["ckd_quick"]

        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 4. Predict
        model=models['ckd_quick']
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
        # 1. Build DataFrame matching original training feature order

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

        # 2. Normalize and Categorical Variables
        le = LabelEncoder()
        df['htn'] = le.fit_transform(df['htn'])
        df['dm'] = le.fit_transform(df['dm'])
        df['cad'] = le.fit_transform(df['cad'])
        df['appet'] = le.fit_transform(df['appet'])
        df['pc'] = le.fit_transform(df['pc'])
        df['pe'] = le.fit_transform(df['pe'])
        df['ane'] = le.fit_transform(df['ane'])

        # 3. Scale numeric features using the pre-fitted scaler
        numeric_cols=['age','bp','sg','al','hemo','pcv','rc','bu','bgr','sc','sod','pot']
        scaler=scalers["ckd_advance"]

        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 4. Predict
        model=models['ckd_advance']
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
    port = int(os.environ.get("PORT", 8800))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)