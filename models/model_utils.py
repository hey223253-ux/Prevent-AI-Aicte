"""
PreventAI – Model Utilities
==============================
Shared utilities for model saving, loading, metrics, and risk classification.
"""

import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


RISK_THRESHOLDS = {
    'Low': (0, 30),
    'Moderate': (30, 60),
    'High': (60, 100),
}


def classify_risk(probability):
    """Convert probability (0-1) to risk percentage and category."""
    risk_pct = round(probability * 100, 1)
    if risk_pct < 30:
        category = 'Low'
    elif risk_pct < 60:
        category = 'Moderate'
    else:
        category = 'High'
    return risk_pct, category


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['classification_report'] = classification_report(
        y_true, y_pred, zero_division=0
    )
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {model_name} – Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['classification_report']}")


def save_sklearn_model(model, path):
    """Save scikit-learn/XGBoost model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  ✓ Saved model to {path}")


def load_sklearn_model(path):
    """Load scikit-learn/XGBoost model."""
    return joblib.load(path)


def save_torch_model(model, path):
    """Save PyTorch model."""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  ✓ Saved PyTorch model to {path}")


def load_torch_model(model_class, path, **kwargs):
    """Load PyTorch model."""
    import torch
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def generate_metrics_report(all_results, output_path=None):
    """
    Generate a markdown comparison table from evaluation results.

    Parameters:
        all_results: dict of {model_name: {target: metrics_dict}}
        output_path: optional path to save markdown file
    """
    lines = ["# PreventAI – Model Evaluation Report\n"]
    lines.append("## Performance Comparison\n")

    targets = ['diabetes_risk', 'cvd_risk', 'hypertension_risk']
    target_names = {
        'diabetes_risk': 'Type 2 Diabetes',
        'cvd_risk': 'Cardiovascular Disease',
        'hypertension_risk': 'Hypertension'
    }

    for target in targets:
        lines.append(f"\n### {target_names.get(target, target)}\n")
        lines.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
        lines.append("|-------|----------|-----------|--------|----------|---------|")

        for model_name, target_results in all_results.items():
            if target in target_results:
                m = target_results[target]
                lines.append(
                    f"| {model_name} | {m['accuracy']:.4f} | "
                    f"{m['precision']:.4f} | {m['recall']:.4f} | "
                    f"{m['f1_score']:.4f} | {m.get('roc_auc', 'N/A'):.4f} |"
                )

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  ✓ Saved evaluation report to {output_path}")

    return report
