"""
PreventAI – Data Preprocessing Module
=======================================
Handles missing values, encoding, scaling, and train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


NUMERIC_FEATURES = [
    'age', 'height_cm', 'weight_kg', 'bmi', 'resting_heart_rate',
    'systolic_bp', 'diastolic_bp', 'daily_steps', 'sleep_duration',
    'stress_score', 'fasting_glucose', 'cholesterol_total',
    'cholesterol_hdl', 'cholesterol_ldl'
]

CATEGORICAL_FEATURES = [
    'gender', 'smoking_status', 'alcohol_consumption'
]

BINARY_FEATURES = [
    'family_history_diabetes', 'family_history_cvd',
    'family_history_hypertension'
]

TARGETS = ['diabetes_risk', 'cvd_risk', 'hypertension_risk']

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def load_data(data_path):
    """Load static patient data from CSV."""
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def handle_missing_values(df):
    """Impute missing values: KNN for numeric, mode for categorical."""
    df = df.copy()

    # KNN imputation for numeric features
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if df[numeric_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"  ✓ KNN-imputed missing values in numeric features")

    # Mode imputation for categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  ✓ Mode-imputed missing values in {col}")

    return df


def encode_categorical(df):
    """One-hot encode all categorical (object/string) features."""
    df = df.copy()
    encoders = {}

    # Find all object/categorical columns (excluding targets and patient_id)
    exclude_cols = TARGETS + ['patient_id']
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                if c not in exclude_cols]

    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # One-hot encode all categorical columns
    if cat_cols:
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    else:
        df_encoded = df

    return df_encoded, encoders


def scale_features(X_train, X_test, feature_cols):
    """Apply standard scaling to numeric features."""
    scaler = StandardScaler()

    numeric_cols = [c for c in NUMERIC_FEATURES if c in feature_cols]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"  ✓ Standard-scaled {len(numeric_cols)} numeric features")
    return X_train, X_test, scaler


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Stratified train-test split."""
    feature_cols = [c for c in df.columns
                    if c not in TARGETS + ['patient_id']]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  ✓ Split: {len(X_train)} train / {len(X_test)} test "
          f"(positive rate: {y_train.mean():.2%} / {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, feature_cols


def preprocess_pipeline(data_path, output_dir=None):
    """
    Full preprocessing pipeline.
    Returns dict of processed data for each target disease.
    """
    print("\n▶ Loading data...")
    df = load_data(data_path)

    print("\n▶ Handling missing values...")
    df = handle_missing_values(df)

    print("\n▶ Encoding categorical variables...")
    df_encoded, encoders = encode_categorical(df)

    results = {}
    for target in TARGETS:
        print(f"\n▶ Preparing data for {target}...")
        X_train, X_test, y_train, y_test, feature_cols = split_data(
            df_encoded, target
        )

        X_train, X_test, scaler = scale_features(X_train, X_test, feature_cols)

        results[target] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'scaler': scaler,
        }

    # Save artifacts
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for target, data in results.items():
            joblib.dump(data['scaler'],
                        os.path.join(output_dir, f'scaler_{target}.pkl'))
        joblib.dump(encoders, os.path.join(output_dir, 'encoders.pkl'))
        print(f"\n  ✓ Saved preprocessing artifacts to {output_dir}")

    return results, encoders


if __name__ == '__main__':
    data_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'static_data.csv'
    )
    results, encoders = preprocess_pipeline(data_file)
    for target, data in results.items():
        print(f"\n{target}: X_train={data['X_train'].shape}, "
              f"X_test={data['X_test'].shape}")
