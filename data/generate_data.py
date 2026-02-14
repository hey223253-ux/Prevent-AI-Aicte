"""
PreventAI – Synthetic Data Generator
=====================================
Generates medically realistic synthetic patient data for disease risk prediction.
Produces:
  - static_data.csv: Patient demographics, vitals, and lab values
  - timeseries_data.csv: 30-day daily lifestyle metrics per patient
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_PATIENTS = 5000
TIMESERIES_DAYS = 30
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def generate_static_data(n=NUM_PATIENTS):
    """Generate static patient features with medically realistic distributions."""
    data = {}

    # Demographics
    data['patient_id'] = np.arange(1, n + 1)
    data['age'] = np.clip(np.random.normal(50, 15, n), 18, 90).astype(int)
    data['gender'] = np.random.choice(['Male', 'Female'], n, p=[0.52, 0.48])

    # Body measurements
    height_m = np.random.normal(1.68, 0.10, n)
    weight_kg = np.random.normal(75, 18, n)
    data['height_cm'] = np.clip(height_m * 100, 140, 210).round(1)
    data['weight_kg'] = np.clip(weight_kg, 40, 180).round(1)
    data['bmi'] = np.clip(weight_kg / (height_m ** 2), 15, 55).round(1)

    # Vitals
    data['resting_heart_rate'] = np.clip(
        np.random.normal(72, 12, n), 45, 110
    ).astype(int)
    data['systolic_bp'] = np.clip(
        np.random.normal(125, 18, n), 85, 200
    ).astype(int)
    data['diastolic_bp'] = np.clip(
        np.random.normal(80, 12, n), 50, 130
    ).astype(int)

    # Lifestyle
    data['daily_steps'] = np.clip(
        np.random.normal(7000, 3500, n), 500, 25000
    ).astype(int)
    data['sleep_duration'] = np.clip(
        np.random.normal(7.0, 1.5, n), 3.0, 12.0
    ).round(1)
    data['smoking_status'] = np.random.choice(
        ['Never', 'Former', 'Current'], n, p=[0.55, 0.25, 0.20]
    )
    data['alcohol_consumption'] = np.random.choice(
        ['None', 'Light', 'Moderate', 'Heavy'], n, p=[0.30, 0.35, 0.25, 0.10]
    )
    data['stress_score'] = np.clip(
        np.random.normal(5, 2.5, n), 0, 10
    ).round(1)

    # Lab values
    data['fasting_glucose'] = np.clip(
        np.random.normal(100, 25, n), 60, 250
    ).round(1)
    data['cholesterol_total'] = np.clip(
        np.random.normal(200, 40, n), 100, 380
    ).round(1)
    data['cholesterol_hdl'] = np.clip(
        np.random.normal(55, 15, n), 20, 100
    ).round(1)
    data['cholesterol_ldl'] = np.clip(
        np.random.normal(120, 35, n), 40, 260
    ).round(1)

    # Family history
    data['family_history_diabetes'] = np.random.choice([0, 1], n, p=[0.65, 0.35])
    data['family_history_cvd'] = np.random.choice([0, 1], n, p=[0.70, 0.30])
    data['family_history_hypertension'] = np.random.choice([0, 1], n, p=[0.60, 0.40])

    df = pd.DataFrame(data)

    # Introduce ~2% missing values in select columns
    missing_cols = [
        'bmi', 'resting_heart_rate', 'fasting_glucose',
        'cholesterol_total', 'sleep_duration', 'daily_steps'
    ]
    for col in missing_cols:
        mask = np.random.random(n) < 0.02
        df.loc[mask, col] = np.nan

    # Generate target labels using probabilistic models
    df = _generate_targets(df)

    return df


def _generate_targets(df):
    """Generate disease risk labels using feature-based probability models."""
    n = len(df)
    df_filled = df.copy()
    for col in df_filled.select_dtypes(include=[np.number]).columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    # --- Diabetes Risk ---
    diabetes_prob = np.zeros(n)
    diabetes_prob += (df_filled['age'] - 30) / 60 * 0.15
    diabetes_prob += (df_filled['bmi'] - 22) / 33 * 0.25
    diabetes_prob += (df_filled['fasting_glucose'] - 70) / 180 * 0.25
    diabetes_prob += df_filled['family_history_diabetes'] * 0.15
    diabetes_prob += (10000 - df_filled['daily_steps']) / 10000 * 0.1
    diabetes_prob += (df_filled['stress_score'] / 10) * 0.05
    diabetes_prob += np.where(df_filled['smoking_status'] == 'Current', 0.05, 0)
    diabetes_prob = np.clip(diabetes_prob + np.random.normal(0, 0.08, n), 0, 1)
    df['diabetes_risk'] = (diabetes_prob > 0.45).astype(int)

    # --- CVD Risk ---
    cvd_prob = np.zeros(n)
    cvd_prob += (df_filled['age'] - 30) / 60 * 0.20
    cvd_prob += (df_filled['systolic_bp'] - 110) / 90 * 0.20
    cvd_prob += (df_filled['cholesterol_total'] - 150) / 230 * 0.15
    cvd_prob += (60 - df_filled['cholesterol_hdl']) / 40 * 0.10
    cvd_prob += df_filled['family_history_cvd'] * 0.12
    cvd_prob += np.where(df_filled['smoking_status'] == 'Current', 0.10, 0)
    cvd_prob += (df_filled['resting_heart_rate'] - 60) / 50 * 0.08
    cvd_prob += (df_filled['bmi'] - 22) / 33 * 0.05
    cvd_prob = np.clip(cvd_prob + np.random.normal(0, 0.08, n), 0, 1)
    df['cvd_risk'] = (cvd_prob > 0.45).astype(int)

    # --- Hypertension Risk ---
    hyper_prob = np.zeros(n)
    hyper_prob += (df_filled['systolic_bp'] - 100) / 100 * 0.30
    hyper_prob += (df_filled['diastolic_bp'] - 60) / 70 * 0.15
    hyper_prob += (df_filled['age'] - 30) / 60 * 0.15
    hyper_prob += (df_filled['bmi'] - 22) / 33 * 0.15
    hyper_prob += df_filled['family_history_hypertension'] * 0.10
    hyper_prob += (df_filled['stress_score'] / 10) * 0.08
    hyper_prob += np.where(df_filled['alcohol_consumption'] == 'Heavy', 0.07, 0)
    hyper_prob = np.clip(hyper_prob + np.random.normal(0, 0.08, n), 0, 1)
    df['hypertension_risk'] = (hyper_prob > 0.45).astype(int)

    return df


def generate_timeseries_data(static_df, days=TIMESERIES_DAYS):
    """Generate 30-day daily lifestyle time-series per patient."""
    records = []
    for _, row in static_df.iterrows():
        pid = int(row['patient_id'])
        base_hr = float(row['resting_heart_rate']) if pd.notna(row.get('resting_heart_rate')) else 72.0
        base_steps = float(row['daily_steps']) if pd.notna(row.get('daily_steps')) else 7000.0
        base_sleep = float(row['sleep_duration']) if pd.notna(row.get('sleep_duration')) else 7.0
        base_stress = float(row['stress_score']) if pd.notna(row.get('stress_score')) else 5.0

        for day in range(1, days + 1):
            records.append({
                'patient_id': pid,
                'day': day,
                'heart_rate': int(np.clip(
                    base_hr + np.random.normal(0, 8), 45, 130
                )),
                'steps': int(np.clip(
                    base_steps + np.random.normal(0, 1500), 0, 30000
                )),
                'sleep_hours': round(float(np.clip(
                    base_sleep + np.random.normal(0, 0.8), 2.0, 14.0
                )), 1),
                'stress_level': round(float(np.clip(
                    base_stress + np.random.normal(0, 1.5), 0, 10
                )), 1),
            })

    return pd.DataFrame(records)


def main():
    """Generate and save all datasets."""
    print("=" * 60)
    print("PreventAI – Synthetic Data Generator")
    print("=" * 60)

    # Static data
    print("\n▶ Generating static patient data...")
    static_df = generate_static_data()
    static_path = os.path.join(OUTPUT_DIR, 'static_data.csv')
    static_df.to_csv(static_path, index=False)
    print(f"  ✓ Saved {len(static_df)} patients to {static_path}")
    print(f"  ✓ Features: {list(static_df.columns)}")
    print(f"  ✓ Target distribution:")
    for target in ['diabetes_risk', 'cvd_risk', 'hypertension_risk']:
        pos = static_df[target].sum()
        print(f"    - {target}: {pos} positive ({pos/len(static_df)*100:.1f}%)")

    # Time-series data
    print("\n▶ Generating time-series lifestyle data...")
    ts_df = generate_timeseries_data(static_df)
    ts_path = os.path.join(OUTPUT_DIR, 'timeseries_data.csv')
    ts_df.to_csv(ts_path, index=False)
    print(f"  ✓ Saved {len(ts_df)} records ({NUM_PATIENTS} patients × {TIMESERIES_DAYS} days)")
    print(f"  ✓ Features: {list(ts_df.columns)}")

    print("\n✅ Data generation complete!")
    return static_df, ts_df


if __name__ == '__main__':
    main()
