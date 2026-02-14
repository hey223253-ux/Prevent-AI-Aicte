"""
PreventAI – SHAP Explainability Module
=========================================
Provides model-agnostic and model-specific explanations
using SHAP (SHapley Additive exPlanations).
"""

import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def get_shap_explainer(model, X_background, model_type='tree'):
    """
    Create appropriate SHAP explainer based on model type.

    Parameters:
        model: trained model
        X_background: background data for KernelExplainer
        model_type: 'tree' for RF/XGBoost, 'linear' for LR, 'kernel' for generic
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_background)
    else:
        # KernelExplainer for any model (slower)
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(
                model.predict_proba, X_background[:100]
            )
        else:
            explainer = shap.KernelExplainer(
                model.predict, X_background[:100]
            )
    return explainer


def get_shap_values(explainer, X, model_type='tree'):
    """Compute SHAP values."""
    shap_values = explainer.shap_values(X)

    # For tree models, shap_values might be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class

    return shap_values


def get_top_risk_factors(shap_values_single, feature_names, top_n=3):
    """
    Get top N contributing risk factors for a single prediction.

    Parameters:
        shap_values_single: SHAP values for one sample
        feature_names: list of feature names
        top_n: number of top factors to return

    Returns:
        List of dicts with 'feature', 'shap_value', 'direction'
    """
    if isinstance(shap_values_single, np.ndarray):
        vals = shap_values_single.flatten()
    else:
        vals = np.array(shap_values_single).flatten()

    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(vals))[::-1][:top_n]

    factors = []
    for idx in indices:
        if idx < len(feature_names):
            factors.append({
                'feature': feature_names[idx],
                'shap_value': float(vals[idx]),
                'direction': 'increases risk' if vals[idx] > 0 else 'decreases risk',
                'importance': float(abs(vals[idx]))
            })

    return factors


def plot_shap_summary(shap_values, X, feature_names, output_path, title="SHAP Summary"):
    """Generate SHAP summary (beeswarm) plot."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved SHAP summary plot: {output_path}")


def plot_shap_bar(shap_values, feature_names, output_path, title="Feature Importance"):
    """Generate SHAP feature importance bar chart."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_idx)))
    y_pos = range(len(sorted_idx))

    ax.barh(y_pos, mean_abs_shap[sorted_idx][::-1],
            color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved SHAP bar plot: {output_path}")


def explain_model(model, X_train, X_test, feature_names, model_name,
                  output_dir, model_type='tree', disease=''):
    """
    Full SHAP explanation pipeline for a model.

    Returns:
        shap_values: SHAP values for test set
        top_factors: top 3 risk factors for first test sample
    """
    print(f"\n▶ Generating SHAP explanations for {model_name} ({disease})...")

    os.makedirs(output_dir, exist_ok=True)

    # Create explainer
    explainer = get_shap_explainer(model, X_train, model_type=model_type)

    # Compute SHAP values
    X_explain = X_test[:200] if len(X_test) > 200 else X_test
    shap_values = get_shap_values(explainer, X_explain, model_type=model_type)

    # Plots
    prefix = f"{model_name.lower().replace(' ', '_')}_{disease}"

    plot_shap_summary(
        shap_values, X_explain, feature_names,
        os.path.join(output_dir, f'shap_summary_{prefix}.png'),
        title=f'{model_name} – {disease} SHAP Summary'
    )

    plot_shap_bar(
        shap_values, feature_names,
        os.path.join(output_dir, f'shap_importance_{prefix}.png'),
        title=f'{model_name} – {disease} Feature Importance'
    )

    # Top factors for first sample
    top_factors = get_top_risk_factors(shap_values[0], feature_names)
    print(f"  Top risk factors (sample 0): {[f['feature'] for f in top_factors]}")

    return shap_values, top_factors
