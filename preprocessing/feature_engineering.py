"""
PreventAI – Feature Engineering Module
========================================
Creates derived features from raw patient and time-series data.
"""

import numpy as np
import pandas as pd


def compute_bmi(df):
    """Calculate BMI from height (cm) and weight (kg) if not present."""
    df = df.copy()
    if 'bmi' not in df.columns and 'height_cm' in df.columns and 'weight_kg' in df.columns:
        df['bmi'] = (df['weight_kg'] / ((df['height_cm'] / 100) ** 2)).round(1)
        print("  ✓ Computed BMI from height and weight")
    return df


def add_bmi_category(df):
    """Categorize BMI into clinical groups."""
    df = df.copy()
    if 'bmi' in df.columns:
        conditions = [
            df['bmi'] < 18.5,
            (df['bmi'] >= 18.5) & (df['bmi'] < 25),
            (df['bmi'] >= 25) & (df['bmi'] < 30),
            df['bmi'] >= 30
        ]
        labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
        df['bmi_category'] = np.select(conditions, labels, default='Unknown')
        print("  ✓ Added BMI category")
    return df


def add_bp_category(df):
    """Classify blood pressure into clinical categories."""
    df = df.copy()
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        conditions = [
            (df['systolic_bp'] < 120) & (df['diastolic_bp'] < 80),
            (df['systolic_bp'] < 130) & (df['diastolic_bp'] < 80),
            (df['systolic_bp'] < 140) | (df['diastolic_bp'] < 90),
            (df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90),
        ]
        labels = ['Normal', 'Elevated', 'High_Stage1', 'High_Stage2']
        df['bp_category'] = np.select(conditions, labels, default='Normal')
        print("  ✓ Added blood pressure category")
    return df


def add_heart_rate_zone(df):
    """Classify resting heart rate into zones."""
    df = df.copy()
    if 'resting_heart_rate' in df.columns:
        conditions = [
            df['resting_heart_rate'] < 60,
            (df['resting_heart_rate'] >= 60) & (df['resting_heart_rate'] < 80),
            (df['resting_heart_rate'] >= 80) & (df['resting_heart_rate'] < 100),
            df['resting_heart_rate'] >= 100,
        ]
        labels = ['Athletic', 'Normal', 'Elevated', 'High']
        df['hr_zone'] = np.select(conditions, labels, default='Normal')
        print("  ✓ Added heart rate zone")
    return df


def add_metabolic_risk_score(df):
    """Compute a composite metabolic risk score (0-10)."""
    df = df.copy()
    score = np.zeros(len(df))

    if 'bmi' in df.columns:
        score += np.clip((df['bmi'].fillna(25) - 18.5) / (40 - 18.5), 0, 1) * 2.5

    if 'fasting_glucose' in df.columns:
        score += np.clip((df['fasting_glucose'].fillna(100) - 70) / (200 - 70), 0, 1) * 2.5

    if 'systolic_bp' in df.columns:
        score += np.clip((df['systolic_bp'].fillna(120) - 90) / (180 - 90), 0, 1) * 2.5

    if 'cholesterol_total' in df.columns:
        score += np.clip((df['cholesterol_total'].fillna(200) - 150) / (300 - 150), 0, 1) * 2.5

    df['metabolic_risk_score'] = score.round(2)
    print("  ✓ Added metabolic risk score")
    return df


def compute_timeseries_features(ts_df):
    """
    Aggregate time-series data into static features per patient.

    Returns DataFrame with:
    - sleep_consistency_score (lower = more consistent)
    - weekly_activity_variance (coefficient of variation of steps)
    - avg_heart_rate, hr_variability
    - avg_stress, stress_variability
    """
    agg = ts_df.groupby('patient_id').agg(
        sleep_consistency=('sleep_hours', 'std'),
        avg_sleep=('sleep_hours', 'mean'),
        weekly_activity_variance=('steps', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        avg_daily_steps=('steps', 'mean'),
        avg_heart_rate=('heart_rate', 'mean'),
        hr_variability=('heart_rate', 'std'),
        avg_stress=('stress_level', 'mean'),
        stress_variability=('stress_level', 'std'),
    ).round(3)

    agg = agg.reset_index()
    print(f"  ✓ Extracted {len(agg.columns) - 1} time-series features "
          f"for {len(agg)} patients")
    return agg


def engineer_features(static_df, ts_df=None):
    """
    Full feature engineering pipeline.

    Parameters:
        static_df: Static patient DataFrame
        ts_df: Optional time-series lifestyle DataFrame

    Returns:
        Enriched DataFrame
    """
    print("\n▶ Feature Engineering...")
    df = static_df.copy()

    df = compute_bmi(df)
    df = add_bmi_category(df)
    df = add_bp_category(df)
    df = add_heart_rate_zone(df)
    df = add_metabolic_risk_score(df)

    if ts_df is not None:
        ts_features = compute_timeseries_features(ts_df)
        df = df.merge(ts_features, on='patient_id', how='left')
        print(f"  ✓ Merged time-series features → {df.shape[1]} total columns")

    return df


if __name__ == '__main__':
    import os
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    static_df = pd.read_csv(os.path.join(data_dir, 'static_data.csv'))
    ts_df = pd.read_csv(os.path.join(data_dir, 'timeseries_data.csv'))

    enriched = engineer_features(static_df, ts_df)
    print(f"\nFinal shape: {enriched.shape}")
    print(f"Columns: {list(enriched.columns)}")
