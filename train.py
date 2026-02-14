"""
PreventAI – Training Orchestrator
====================================
End-to-end training pipeline: data generation → preprocessing →
feature engineering → model training → evaluation → visualization → fairness.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.generate_data import main as generate_data
from preprocessing.feature_engineering import engineer_features
from preprocessing.preprocess import (
    handle_missing_values, encode_categorical, scale_features,
    split_data, NUMERIC_FEATURES, TARGETS
)
from models.baseline_models import train_all_baseline_models, predict_with_model
from models.model_utils import compute_metrics, print_metrics, save_sklearn_model
from evaluation.evaluate import compare_models
from evaluation.visualize import generate_all_visualizations
from explainability.shap_explainer import explain_model
from ethics.fairness import generate_fairness_report


# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

DISEASE_NAMES = {
    'diabetes_risk': 'Type 2 Diabetes',
    'cvd_risk': 'Cardiovascular Disease',
    'hypertension_risk': 'Hypertension'
}


def setup_dirs():
    """Create output directories."""
    for d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def step_generate_data():
    """Step 1: Generate synthetic data."""
    print("\n" + "=" * 70)
    print("  STEP 1: DATA GENERATION")
    print("=" * 70)
    static_df, ts_df = generate_data()
    return static_df, ts_df


def step_feature_engineering(static_df, ts_df):
    """Step 2: Feature engineering."""
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    enriched_df = engineer_features(static_df, ts_df)
    return enriched_df


def step_preprocess_and_split(df, target):
    """Step 3: Preprocessing and splitting for a target."""
    print(f"\n  Preprocessing for {DISEASE_NAMES.get(target, target)}...")

    # Handle missing values
    df_clean = handle_missing_values(df)

    # Encode categorical
    df_encoded, encoders = encode_categorical(df_clean)

    # Split
    X_train, X_test, y_train, y_test, feature_cols = split_data(df_encoded, target)

    # Scale
    X_train, X_test, scaler = scale_features(X_train, X_test, feature_cols)

    return X_train, X_test, y_train, y_test, feature_cols, scaler, encoders


def step_train_and_evaluate(X_train, X_test, y_train, y_test,
                             feature_cols, target, patient_data):
    """Step 4: Train models and evaluate."""
    print(f"\n{'='*70}")
    print(f"  STEP 4: TRAINING & EVALUATION – {DISEASE_NAMES.get(target, target)}")
    print(f"{'='*70}")

    # Train baseline models
    models = train_all_baseline_models(X_train, y_train, tune_rf=True)

    # Evaluate each model
    eval_results = {}
    all_metrics = {}

    for name, model in models.items():
        y_pred, y_prob = predict_with_model(model, X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, f"{name} ({DISEASE_NAMES.get(target, target)})")

        eval_results[name] = {
            'model': model,
            'metrics': metrics,
            'y_true': y_test.values,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'feature_names': feature_cols,
        }
        all_metrics[name] = metrics

        # Save model
        model_filename = f"{name.lower().replace(' ', '_')}_{target}.pkl"
        save_sklearn_model(model, os.path.join(MODELS_DIR, model_filename))

    return models, eval_results, all_metrics


def step_visualize(eval_results, target):
    """Step 5: Generate visualizations."""
    print(f"\n{'='*70}")
    print(f"  STEP 5: VISUALIZATION – {DISEASE_NAMES.get(target, target)}")
    print(f"{'='*70}")

    generate_all_visualizations(
        eval_results,
        PLOTS_DIR,
        disease_name=DISEASE_NAMES.get(target, target)
    )


def step_explainability(models, X_train, X_test, feature_cols, target):
    """Step 6: SHAP explainability."""
    print(f"\n{'='*70}")
    print(f"  STEP 6: EXPLAINABILITY – {DISEASE_NAMES.get(target, target)}")
    print(f"{'='*70}")

    # Use XGBoost for SHAP (best support)
    if 'XGBoost' in models:
        explain_model(
            models['XGBoost'], X_train, X_test, feature_cols,
            model_name='XGBoost',
            output_dir=PLOTS_DIR,
            model_type='tree',
            disease=DISEASE_NAMES.get(target, target)
        )

    if 'Random Forest' in models:
        explain_model(
            models['Random Forest'], X_train, X_test, feature_cols,
            model_name='Random Forest',
            output_dir=PLOTS_DIR,
            model_type='tree',
            disease=DISEASE_NAMES.get(target, target)
        )


def step_fairness(eval_results, patient_data, target):
    """Step 7: Fairness evaluation."""
    print(f"\n{'='*70}")
    print(f"  STEP 7: FAIRNESS EVALUATION – {DISEASE_NAMES.get(target, target)}")
    print(f"{'='*70}")

    # Use best model (XGBoost)
    best_model_name = max(
        eval_results.keys(),
        key=lambda k: eval_results[k]['metrics'].get('roc_auc', 0)
    )
    best = eval_results[best_model_name]

    generate_fairness_report(
        y_true=best['y_true'],
        y_pred=best['y_pred'],
        y_prob=best['y_prob'],
        patient_data=patient_data,
        model_name=best_model_name,
        disease_name=DISEASE_NAMES.get(target, target),
        output_dir=REPORTS_DIR
    )


def step_train_lstm(enriched_df, ts_df, target, feature_cols, X_train, X_test, y_train, y_test):
    """Step 8: Train LSTM model on time-series data."""
    print(f"\n{'='*70}")
    print(f"  STEP 8: LSTM TRAINING – {DISEASE_NAMES.get(target, target)}")
    print(f"{'='*70}")

    try:
        from models.lstm_model import prepare_lstm_data, train_lstm_model, predict_lstm, LSTMRiskPredictor
        from models.model_utils import save_torch_model
        import torch

        # Prepare LSTM data
        train_ids = X_train.index if hasattr(X_train, 'index') else range(len(X_train))
        test_ids = X_test.index if hasattr(X_test, 'index') else range(len(X_test))

        train_static_df = enriched_df.loc[train_ids].reset_index(drop=True)
        test_static_df = enriched_df.loc[test_ids].reset_index(drop=True)

        # Use feature_cols that exist in static_df for LSTM static input
        lstm_feature_cols = [c for c in feature_cols if c in enriched_df.columns]

        train_static, train_ts, train_labels, ts_scaler = prepare_lstm_data(
            train_static_df, ts_df, target, lstm_feature_cols
        )
        test_static, test_ts, test_labels, _ = prepare_lstm_data(
            test_static_df, ts_df, target, lstm_feature_cols
        )

        # Train LSTM
        print(f"\n  Training LSTM model...")
        model, history = train_lstm_model(
            train_static, train_ts, train_labels,
            val_static=test_static, val_ts=test_ts, val_labels=test_labels,
            epochs=30, batch_size=64, lr=0.001
        )

        # Evaluate
        y_pred_lstm, y_prob_lstm = predict_lstm(model, test_static, test_ts)
        lstm_metrics = compute_metrics(test_labels, y_pred_lstm, y_prob_lstm)
        print_metrics(lstm_metrics, f"LSTM ({DISEASE_NAMES.get(target, target)})")

        # Save model
        save_torch_model(model, os.path.join(MODELS_DIR, f'lstm_{target}.pt'))

        return lstm_metrics, model

    except Exception as e:
        print(f"  ⚠️ LSTM training failed: {e}")
        print(f"  Continuing with baseline models only.")
        return None, None


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='PreventAI – Training Pipeline')
    parser.add_argument('--targets', nargs='+',
                        default=['diabetes_risk', 'cvd_risk', 'hypertension_risk'],
                        help='Disease targets to train')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM training')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP explainability')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: no tuning, no LSTM')
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  PreventAI – Early Disease Risk Prediction System")
    print("  Training Pipeline")
    print("█" * 70)

    setup_dirs()

    # Step 1: Generate data
    static_df, ts_df = step_generate_data()

    # Step 2: Feature engineering
    enriched_df = step_feature_engineering(static_df, ts_df)

    # Save enriched data
    enriched_path = os.path.join(OUTPUT_DIR, 'enriched_data.csv')
    enriched_df.to_csv(enriched_path, index=False)
    print(f"\n  ✓ Saved enriched data to {enriched_path}")

    # Process each disease target
    all_results = {}

    for target in args.targets:
        print(f"\n\n{'#'*70}")
        print(f"  PROCESSING: {DISEASE_NAMES.get(target, target).upper()}")
        print(f"{'#'*70}")

        # Step 3: Preprocess
        X_train, X_test, y_train, y_test, feature_cols, scaler, encoders = \
            step_preprocess_and_split(enriched_df, target)

        # Save preprocessing artifacts
        joblib.dump(scaler, os.path.join(MODELS_DIR, f'scaler_{target}.pkl'))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, f'feature_cols_{target}.pkl'))
        joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))

        # Step 4: Train & evaluate baseline models
        models, eval_results, target_metrics = step_train_and_evaluate(
            X_train, X_test, y_train, y_test, feature_cols, target, enriched_df
        )

        # Step 5: Visualizations
        step_visualize(eval_results, target)

        # Step 6: SHAP explainability
        if not args.skip_shap and not args.quick:
            step_explainability(models, X_train, X_test, feature_cols, target)

        # Step 7: Fairness
        step_fairness(eval_results, enriched_df, target)

        # Step 8: LSTM (optional)
        if not args.skip_lstm and not args.quick:
            lstm_metrics, lstm_model = step_train_lstm(
                enriched_df, ts_df, target, feature_cols,
                X_train, X_test, y_train, y_test
            )
            if lstm_metrics:
                target_metrics['LSTM'] = lstm_metrics

        # Collect results
        for model_name, metrics in target_metrics.items():
            if model_name not in all_results:
                all_results[model_name] = {}
            all_results[model_name][target] = metrics

    # Final comparison report
    print(f"\n\n{'='*70}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'='*70}")

    report = compare_models(
        all_results,
        output_path=os.path.join(REPORTS_DIR, 'evaluation_report.md')
    )
    print(report)

    # Summary
    print(f"\n\n{'█'*70}")
    print("  ✅ PreventAI Training Pipeline Complete!")
    print(f"{'█'*70}")
    print(f"\n  📁 Outputs saved to: {OUTPUT_DIR}")
    print(f"     ├── models/     → Trained model files")
    print(f"     ├── plots/      → Visualization charts")
    print(f"     └── reports/    → Evaluation & fairness reports")
    print(f"\n  🚀 To start the API:")
    print(f"     cd \"{PROJECT_ROOT}\"")
    print(f"     python -m uvicorn api.app:app --reload")
    print(f"\n  🌐 To start the Streamlit dashboard:")
    print(f"     streamlit run frontend/streamlit_app.py")
    print(f"\n  📖 API docs: http://localhost:8000/docs")


if __name__ == '__main__':
    main()
