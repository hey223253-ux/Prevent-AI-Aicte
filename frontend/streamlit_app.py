"""
PreventAI – Streamlit Frontend
=================================
Interactive dashboard for disease risk prediction.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# Page config
st.set_page_config(
    page_title="PreventAI – Disease Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: #F5F7FA;
        color: #1a1a1a;
    }
    /* Force all text to dark color */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp strong, .stApp em {
        color: #1a1a1a !important;
    }
    .stApp .stMetricValue, .stApp .stMetricLabel {
        color: #1a1a1a !important;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #4A90D9, #7B68EE);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(74, 144, 217, 0.25);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    .risk-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        color: #1a1a1a !important;
    }
    .risk-card strong, .risk-card span {
        color: #1a1a1a !important;
    }
    .risk-low { border-left: 4px solid #2ecc71; }
    .risk-moderate { border-left: 4px solid #f39c12; }
    .risk-high { border-left: 4px solid #e74c3c; }
    .disclaimer {
        background: #FFF3F0;
        border: 1px solid #FDDDD6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #7a2e0e !important;
    }
    .disclaimer strong {
        color: #c0392b !important;
    }
    div[data-testid="stSidebar"] {
        background: #EEF1F6;
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] div,
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# API config
API_URL = os.environ.get("PREVENTAI_API_URL", "http://localhost:8000")


def create_gauge_chart(value, title, max_val=100):
    """Create a gauge chart for risk visualization."""
    if value < 30:
        color = "#2ecc71"
        bar_color = "rgba(46, 204, 113, 0.3)"
    elif value < 60:
        color = "#f39c12"
        bar_color = "rgba(243, 156, 18, 0.3)"
    else:
        color = "#e74c3c"
        bar_color = "rgba(231, 76, 60, 0.3)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={'suffix': '%', 'font': {'size': 40, 'color': '#1a1a1a'}},
        title={'text': title, 'font': {'size': 16, 'color': '#333333'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#666666'},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.15)'},
                {'range': [30, 60], 'color': 'rgba(243, 156, 18, 0.15)'},
                {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.15)'},
            ],
            'threshold': {
                'line': {'color': '#333333', 'width': 3},
                'thickness': 0.8,
                'value': value
            },
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1a1a1a'}
    )
    return fig


def create_risk_factors_chart(factors):
    """Create horizontal bar chart for risk factors."""
    if not factors:
        return None

    names = [f['feature'].replace('_', ' ').title() for f in factors]
    values = [f['importance'] for f in factors]
    colors = ['#e94560' if f['direction'] == 'increases risk' else '#48c9b0' for f in factors]

    fig = go.Figure(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f['direction'] for f in factors],
        textposition='auto',
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1a1a1a', 'size': 11},
        xaxis={'title': 'Importance', 'color': '#1a1a1a', 'gridcolor': 'rgba(0,0,0,0.1)'},
        yaxis={'color': '#1a1a1a'},
    )
    return fig


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 PreventAI</h1>
        <p>Early Disease Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This system is for research
        purposes. Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar – Patient Input Form
    with st.sidebar:
        st.markdown("## 📋 Patient Information")

        st.markdown("### Demographics")
        age = st.slider("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("### Body Measurements")
        height_cm = st.number_input("Height (cm)", 100.0, 250.0, 170.0, step=1.0)
        weight_kg = st.number_input("Weight (kg)", 30.0, 250.0, 75.0, step=1.0)
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        st.metric("Calculated BMI", bmi)

        st.markdown("### Vital Signs")
        resting_hr = st.slider("Resting Heart Rate (bpm)", 40, 150, 72)
        systolic_bp = st.slider("Systolic BP (mmHg)", 70, 250, 120)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 40, 150, 80)

        st.markdown("### Lifestyle")
        daily_steps = st.slider("Daily Steps", 0, 30000, 7000, step=500)
        sleep_duration = st.slider("Sleep Duration (hrs)", 1.0, 16.0, 7.0, step=0.5)
        stress_score = st.slider("Stress Score (0-10)", 0.0, 10.0, 5.0, step=0.5)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])

        st.markdown("### Lab Values")
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", 50.0, 400.0, 100.0)
        cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", 80.0, 500.0, 200.0)
        cholesterol_hdl = st.number_input("HDL Cholesterol (mg/dL)", 10.0, 150.0, 55.0)
        cholesterol_ldl = st.number_input("LDL Cholesterol (mg/dL)", 20.0, 350.0, 120.0)

        st.markdown("### Family History")
        fh_diabetes = st.checkbox("Family History – Diabetes")
        fh_cvd = st.checkbox("Family History – Cardiovascular Disease")
        fh_hypertension = st.checkbox("Family History – Hypertension")

        predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    # Main content
    if predict_btn:
        payload = {
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "resting_heart_rate": resting_hr,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "daily_steps": daily_steps,
            "sleep_duration": sleep_duration,
            "fasting_glucose": fasting_glucose,
            "cholesterol_total": cholesterol_total,
            "cholesterol_hdl": cholesterol_hdl,
            "cholesterol_ldl": cholesterol_ldl,
            "smoking_status": smoking,
            "alcohol_consumption": alcohol,
            "stress_score": stress_score,
            "family_history_diabetes": int(fh_diabetes),
            "family_history_cvd": int(fh_cvd),
            "family_history_hypertension": int(fh_hypertension),
        }

        try:
            with st.spinner("🔄 Analyzing risk factors..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                # Patient Summary
                st.markdown("### 👤 Patient Summary")
                summary = result.get('patient_summary', {})
                cols = st.columns(5)
                cols[0].metric("Age", summary.get('age', 'N/A'))
                cols[1].metric("Gender", summary.get('gender', 'N/A'))
                cols[2].metric("BMI", summary.get('bmi', 'N/A'))
                cols[3].metric("Blood Pressure", summary.get('blood_pressure', 'N/A'))
                cols[4].metric("Fasting Glucose", summary.get('fasting_glucose', 'N/A'))

                st.markdown("---")
                st.markdown("### 📊 Risk Assessment Results")

                # Gauges
                predictions = result.get('predictions', [])
                gauge_cols = st.columns(len(predictions))

                for i, pred in enumerate(predictions):
                    with gauge_cols[i]:
                        fig = create_gauge_chart(
                            pred['risk_probability_pct'],
                            pred['disease']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Risk category badge
                        cat = pred['risk_category']
                        cat_class = f"risk-{cat.lower()}"
                        st.markdown(
                            f'<div class="risk-card {cat_class}">'
                            f'<strong>Category:</strong> {cat}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Risk Factors
                st.markdown("---")
                st.markdown("### 🔑 Top Risk Factors")

                factor_cols = st.columns(len(predictions))
                for i, pred in enumerate(predictions):
                    with factor_cols[i]:
                        st.markdown(f"**{pred['disease']}**")
                        factors = pred.get('top_risk_factors', [])
                        if factors:
                            fig = create_risk_factors_chart(factors)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No risk factors available")

            elif response.status_code == 503:
                st.error("⚠️ Models not loaded. Please run `python train.py` first to train the models.")
            else:
                st.error(f"API Error: {response.status_code} – {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to the API server. "
                "Please start it with: `python -m uvicorn api.app:app --reload`"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        # Default view
        st.markdown("""
        ### Welcome to PreventAI 👋

        This system uses machine learning to predict your **3–5 year risk** for:

        - 🩸 **Type 2 Diabetes**
        - ❤️ **Cardiovascular Disease**
        - 💉 **Hypertension**

        #### How to Use
        1. Fill in your health information in the sidebar
        2. Click **🔍 Predict Risk**
        3. Review your risk assessment and top risk factors

        #### Technology
        - **Models**: Logistic Regression, Random Forest, XGBoost, LSTM
        - **Explainability**: SHAP (SHapley Additive exPlanations)
        - **Fairness**: Evaluated across age and gender groups
        """)

        # Show sample gauge charts
        st.markdown("### Sample Risk Visualization")
        demo_cols = st.columns(3)
        demo_data = [
            ("Type 2 Diabetes", 25),
            ("Cardiovascular Disease", 52),
            ("Hypertension", 78),
        ]
        for col, (name, val) in zip(demo_cols, demo_data):
            with col:
                fig = create_gauge_chart(val, name)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
