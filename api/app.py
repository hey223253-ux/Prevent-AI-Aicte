"""
PreventAI – FastAPI Backend
==============================
REST API for disease risk prediction with JSON input/output.
"""

import os
import sys
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.model_utils import classify_risk

app = FastAPI(
    title="PreventAI – Disease Risk Prediction API",
    description=(
        "Predicts 3–5 year risk probability for Type 2 Diabetes, "
        "Cardiovascular Disease, and Hypertension.\n\n"
        "⚠️ **MEDICAL DISCLAIMER**: This system is for educational and research "
        "purposes only. It is NOT a substitute for professional medical advice, "
        "diagnosis, or treatment. Always seek the advice of your physician or "
        "other qualified health provider."
    ),
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class PatientInput(BaseModel):
    """Patient data input schema."""
    age: int = Field(..., ge=18, le=100, description="Patient age (18-100)")
    gender: str = Field(..., description="Patient gender (Male/Female)")
    height_cm: float = Field(..., ge=100, le=250, description="Height in cm")
    weight_kg: float = Field(..., ge=30, le=250, description="Weight in kg")
    bmi: Optional[float] = Field(None, description="BMI (auto-calculated if omitted)")
    resting_heart_rate: int = Field(..., ge=40, le=150, description="Resting heart rate (bpm)")
    systolic_bp: int = Field(..., ge=70, le=250, description="Systolic blood pressure")
    diastolic_bp: int = Field(..., ge=40, le=150, description="Diastolic blood pressure")
    daily_steps: int = Field(..., ge=0, le=50000, description="Average daily steps")
    sleep_duration: float = Field(..., ge=1, le=16, description="Average sleep hours")
    fasting_glucose: float = Field(..., ge=50, le=400, description="Fasting glucose (mg/dL)")
    cholesterol_total: float = Field(..., ge=80, le=500, description="Total cholesterol (mg/dL)")
    cholesterol_hdl: float = Field(..., ge=10, le=150, description="HDL cholesterol (mg/dL)")
    cholesterol_ldl: float = Field(..., ge=20, le=350, description="LDL cholesterol (mg/dL)")
    smoking_status: str = Field(..., description="Smoking status (Never/Former/Current)")
    alcohol_consumption: str = Field(..., description="Alcohol (None/Light/Moderate/Heavy)")
    stress_score: float = Field(..., ge=0, le=10, description="Stress score (0-10)")
    family_history_diabetes: int = Field(..., ge=0, le=1, description="Family history of diabetes (0/1)")
    family_history_cvd: int = Field(..., ge=0, le=1, description="Family history of CVD (0/1)")
    family_history_hypertension: int = Field(..., ge=0, le=1, description="Family history of hypertension (0/1)")


class RiskFactor(BaseModel):
    feature: str
    importance: float
    direction: str


class DiseaseRisk(BaseModel):
    disease: str
    risk_probability_pct: float
    risk_category: str
    top_risk_factors: List[RiskFactor]


class PredictionResponse(BaseModel):
    patient_summary: Dict
    predictions: List[DiseaseRisk]
    medical_disclaimer: str


# --- Model Loading ---

MODELS = {}
SCALERS = {}
ENCODERS = None
FEATURE_COLUMNS = {}


def load_models():
    """Load all trained models and preprocessing artifacts."""
    global MODELS, SCALERS, ENCODERS, FEATURE_COLUMNS

    models_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    if not os.path.exists(models_dir):
        print("⚠️ No trained models found. Run train.py first.")
        return False

    targets = ['diabetes_risk', 'cvd_risk', 'hypertension_risk']

    for target in targets:
        # Load best model (XGBoost by default)
        model_path = os.path.join(models_dir, f'xgboost_{target}.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f'random_forest_{target}.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f'logistic_regression_{target}.pkl')

        if os.path.exists(model_path):
            MODELS[target] = joblib.load(model_path)
            print(f"  ✓ Loaded model for {target}")

        # Load scaler
        scaler_path = os.path.join(models_dir, f'scaler_{target}.pkl')
        if os.path.exists(scaler_path):
            SCALERS[target] = joblib.load(scaler_path)

        # Load feature columns
        cols_path = os.path.join(models_dir, f'feature_cols_{target}.pkl')
        if os.path.exists(cols_path):
            FEATURE_COLUMNS[target] = joblib.load(cols_path)

    # Load encoders
    encoder_path = os.path.join(models_dir, 'encoders.pkl')
    if os.path.exists(encoder_path):
        ENCODERS = joblib.load(encoder_path)

    return len(MODELS) > 0


def prepare_input(patient: PatientInput, target: str):
    """Transform patient input into model-ready features."""
    import pandas as pd

    # Calculate BMI if not provided
    bmi = patient.bmi
    if bmi is None:
        bmi = round(patient.weight_kg / ((patient.height_cm / 100) ** 2), 1)

    data = {
        'age': patient.age,
        'height_cm': patient.height_cm,
        'weight_kg': patient.weight_kg,
        'bmi': bmi,
        'resting_heart_rate': patient.resting_heart_rate,
        'systolic_bp': patient.systolic_bp,
        'diastolic_bp': patient.diastolic_bp,
        'daily_steps': patient.daily_steps,
        'sleep_duration': patient.sleep_duration,
        'fasting_glucose': patient.fasting_glucose,
        'cholesterol_total': patient.cholesterol_total,
        'cholesterol_hdl': patient.cholesterol_hdl,
        'cholesterol_ldl': patient.cholesterol_ldl,
        'stress_score': patient.stress_score,
        'family_history_diabetes': patient.family_history_diabetes,
        'family_history_cvd': patient.family_history_cvd,
        'family_history_hypertension': patient.family_history_hypertension,
    }

    df = pd.DataFrame([data])

    # One-hot encode categorical features
    df['gender'] = patient.gender
    df['smoking_status'] = patient.smoking_status
    df['alcohol_consumption'] = patient.alcohol_consumption
    df = pd.get_dummies(df, columns=['gender', 'smoking_status', 'alcohol_consumption'])

    # Align with training feature columns
    if target in FEATURE_COLUMNS:
        for col in FEATURE_COLUMNS[target]:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_COLUMNS[target]]

    # Scale numeric features
    if target in SCALERS:
        from preprocessing.preprocess import NUMERIC_FEATURES
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if numeric_cols:
            df[numeric_cols] = SCALERS[target].transform(df[numeric_cols])

    return df


def get_top_factors(model, features_df, feature_names, top_n=3):
    """Get top risk factors using feature importances or coefficients."""
    factors = []

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return factors

    values = features_df.values[0] if len(features_df) > 0 else np.zeros(len(feature_names))
    top_idx = np.argsort(importances)[::-1][:top_n]

    for idx in top_idx:
        if idx < len(feature_names):
            direction = 'increases risk'
            if hasattr(model, 'coef_') and model.coef_[0][idx] < 0:
                direction = 'decreases risk'
            factors.append(RiskFactor(
                feature=feature_names[idx],
                importance=round(float(importances[idx]), 4),
                direction=direction
            ))

    return factors


# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    """Load models on server startup."""
    print("\n🚀 PreventAI API Starting...")
    success = load_models()
    if success:
        print(f"  ✓ Loaded {len(MODELS)} disease models")
    else:
        print("  ⚠️ No models loaded. Predictions will fail until models are trained.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "available_predictions": list(MODELS.keys()),
        "disclaimer": (
            "This is an educational ML system. "
            "Do NOT use for actual medical decisions."
        )
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """
    Predict disease risk for a patient.

    Returns risk probability, category, and top 3 risk factors
    for each disease (Diabetes, CVD, Hypertension).
    """
    if not MODELS:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Please run train.py first."
        )

    disease_names = {
        'diabetes_risk': 'Type 2 Diabetes',
        'cvd_risk': 'Cardiovascular Disease',
        'hypertension_risk': 'Hypertension'
    }

    predictions = []

    for target, model in MODELS.items():
        try:
            features_df = prepare_input(patient, target)
            feature_names = list(features_df.columns)

            prob = model.predict_proba(features_df)[0][1]
            risk_pct, risk_category = classify_risk(prob)

            top_factors = get_top_factors(model, features_df, feature_names)

            predictions.append(DiseaseRisk(
                disease=disease_names.get(target, target),
                risk_probability_pct=risk_pct,
                risk_category=risk_category,
                top_risk_factors=top_factors
            ))
        except Exception as e:
            predictions.append(DiseaseRisk(
                disease=disease_names.get(target, target),
                risk_probability_pct=0.0,
                risk_category="Error",
                top_risk_factors=[]
            ))
            print(f"  ⚠️ Error predicting {target}: {e}")

    bmi = patient.bmi or round(patient.weight_kg / ((patient.height_cm / 100) ** 2), 1)

    return PredictionResponse(
        patient_summary={
            "age": patient.age,
            "gender": patient.gender,
            "bmi": bmi,
            "blood_pressure": f"{patient.systolic_bp}/{patient.diastolic_bp}",
            "fasting_glucose": patient.fasting_glucose,
        },
        predictions=predictions,
        medical_disclaimer=(
            "⚠️ IMPORTANT: This prediction is generated by a machine learning model "
            "for educational purposes only. It is NOT a medical diagnosis. "
            "Please consult a qualified healthcare professional for medical advice. "
            "Do not make health decisions based solely on this output."
        )
    )


@app.get("/")
async def root():
    return {
        "name": "PreventAI – Early Disease Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST – Submit patient data for risk prediction",
            "/health": "GET – System health check",
            "/docs": "GET – Interactive API documentation (Swagger UI)",
        },
        "disclaimer": (
            "For educational and research purposes only. "
            "Not intended for clinical use."
        )
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
