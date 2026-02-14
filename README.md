# PreventAI – Early Disease Risk Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **Medical Disclaimer**: This system is for **educational and research purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## Overview

**PreventAI** is a machine learning system that predicts **3–5 year risk probability** for three major chronic diseases:

| Disease | Description |
|---------|-------------|
| 🩸 **Type 2 Diabetes** | Based on metabolic, lifestyle, and genetic risk factors |
| ❤️ **Cardiovascular Disease** | Using cardiac vitals, cholesterol, and behavioral data |
| 💉 **Hypertension** | From blood pressure trends, stress, and demographics |

## Architecture

```
PreventAI/
├── data/                        # Data generation
│   └── generate_data.py         # Synthetic dataset generator (5,000 patients)
├── preprocessing/               # Data pipeline
│   ├── preprocess.py            # Missing values, encoding, scaling
│   └── feature_engineering.py   # BMI, sleep score, metabolic risk
├── models/                      # ML models
│   ├── baseline_models.py       # Logistic Regression, Random Forest, XGBoost
│   ├── lstm_model.py            # PyTorch LSTM for time-series data
│   └── model_utils.py           # Metrics, saving, risk classification
├── explainability/              # Model interpretability
│   └── shap_explainer.py        # SHAP analysis & top risk factors
├── evaluation/                  # Model assessment
│   ├── evaluate.py              # Cross-validation & comparison reports
│   └── visualize.py             # ROC curves, confusion matrices, charts
├── ethics/                      # Fairness & bias
│   └── fairness.py              # Demographic parity, equalized odds
├── api/                         # REST API
│   └── app.py                   # FastAPI backend (/predict endpoint)
├── frontend/                    # Web UI
│   └── streamlit_app.py         # Interactive Streamlit dashboard
├── train.py                     # Training orchestrator
├── requirements.txt             # Python dependencies
└── outputs/                     # Generated outputs
    ├── models/                  # Saved model files
    ├── plots/                   # Visualization charts
    └── reports/                 # Evaluation & fairness reports
```

## Features Used

| Category | Features |
|----------|----------|
| **Demographics** | Age, Gender |
| **Body Metrics** | BMI, Height, Weight |
| **Vitals** | Resting Heart Rate, Systolic/Diastolic BP |
| **Lifestyle** | Daily Steps, Sleep Duration, Smoking, Alcohol, Stress Score |
| **Lab Values** | Fasting Glucose, Total/HDL/LDL Cholesterol |
| **Family History** | Diabetes, CVD, Hypertension |
| **Engineered** | BMI Category, BP Category, HR Zone, Metabolic Risk Score, Sleep Consistency, Activity Variance |

## Models

### Baseline Models
- **Logistic Regression** – Linear classifier with class balancing
- **Random Forest** – Ensemble with GridSearchCV hyperparameter tuning
- **XGBoost** – Gradient-boosted trees with early stopping

### Advanced Model
- **LSTM (Long Short-Term Memory)** – PyTorch recurrent neural network processing 30-day lifestyle time-series data (heart rate, steps, sleep, stress) combined with static patient features

### Explainability
- **SHAP** (SHapley Additive exPlanations) for model-agnostic feature importance and per-patient top-3 risk factor identification

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Full pipeline (data generation → training → evaluation → SHAP → fairness)
python train.py

# Quick mode (no LSTM, no hyperparameter tuning)
python train.py --quick

# Train for specific diseases only
python train.py --targets diabetes_risk cvd_risk

# Skip LSTM training
python train.py --skip-lstm
```

### 3. Start the API

```bash
python -m uvicorn api.app:app --reload --port 8000
```

API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start the Dashboard

```bash
streamlit run frontend/streamlit_app.py
```

## API Usage

### POST `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "Male",
    "height_cm": 175,
    "weight_kg": 85,
    "resting_heart_rate": 78,
    "systolic_bp": 145,
    "diastolic_bp": 92,
    "daily_steps": 4000,
    "sleep_duration": 5.5,
    "fasting_glucose": 126,
    "cholesterol_total": 240,
    "cholesterol_hdl": 38,
    "cholesterol_ldl": 165,
    "smoking_status": "Current",
    "alcohol_consumption": "Moderate",
    "stress_score": 7.5,
    "family_history_diabetes": 1,
    "family_history_cvd": 1,
    "family_history_hypertension": 0
  }'
```

### Response Example

```json
{
  "patient_summary": {
    "age": 55,
    "gender": "Male",
    "bmi": 27.8,
    "blood_pressure": "145/92",
    "fasting_glucose": 126.0
  },
  "predictions": [
    {
      "disease": "Type 2 Diabetes",
      "risk_probability_pct": 72.3,
      "risk_category": "High",
      "top_risk_factors": [
        {"feature": "fasting_glucose", "importance": 0.245, "direction": "increases risk"},
        {"feature": "bmi", "importance": 0.189, "direction": "increases risk"},
        {"feature": "family_history_diabetes", "importance": 0.156, "direction": "increases risk"}
      ]
    }
  ],
  "medical_disclaimer": "⚠️ IMPORTANT: This prediction is generated by a machine learning model..."
}
```

## Evaluation Metrics

Models are evaluated using:
- **Accuracy** – Overall correctness
- **Precision** – Positive predictive value
- **Recall** – Sensitivity / True Positive Rate
- **F1-Score** – Harmonic mean of precision and recall
- **ROC-AUC** – Area Under the ROC Curve

## Ethical Considerations

### Fairness Evaluation
- Model performance analyzed across **age groups** (18-30, 31-45, 46-60, 61-75, 75+) and **gender**
- **Equalized Odds** and **Demographic Parity** metrics computed
- Fairness reports generated in `outputs/reports/`

### Bias Discussion
- Training data limitations acknowledged
- Feature representation biases documented
- Mitigation strategies recommended (adversarial debiasing, calibration, human-in-the-loop)

### Medical Safety
- Prominent disclaimers in API responses and UI
- System explicitly designed as educational tool, NOT clinical decision support
- Recommendations to always consult healthcare professionals

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## License

This project is for educational purposes. See [LICENSE](LICENSE) for details.
