"""
PreventAI – Visualization Module
===================================
Generates plots for model evaluation: ROC curves, confusion matrices,
feature importance charts, and risk distribution histograms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os


# Style configuration
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'grid.color': '#333',
    'font.family': 'sans-serif',
    'font.size': 11,
})

COLORS = ['#e94560', '#0f3460', '#533483', '#48c9b0', '#f39c12', '#3498db']


def plot_roc_curves(models_data, output_path, title="ROC Curves"):
    """
    Plot overlaid ROC curves for multiple models.

    Parameters:
        models_data: dict of {model_name: (y_true, y_prob)}
        output_path: file path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, (y_true, y_prob)) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = COLORS[i % len(COLORS)]
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold', color='white')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved ROC curve: {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Plot a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='RdPu',
        ax=ax, cbar_kws={'shrink': 0.8},
        annot_kws={'size': 16, 'fontweight': 'bold'},
        linewidths=2, linecolor='#1a1a2e'
    )
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticklabels(['No Risk', 'At Risk'], fontsize=11)
    ax.set_yticklabels(['No Risk', 'At Risk'], fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved confusion matrix: {output_path}")


def plot_model_comparison(all_results, output_path, disease_name=""):
    """
    Bar chart comparing model metrics.

    Parameters:
        all_results: dict of {model_name: metrics_dict}
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        values = [all_results[model_name].get(m, 0) for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9,
                      label=model_name, color=COLORS[i % len(COLORS)],
                      edgecolor='white', linewidth=0.5, alpha=0.9)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#eee')

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title(f'Model Performance Comparison – {disease_name}',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=11, framealpha=0.8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved model comparison: {output_path}")


def plot_risk_distribution(probabilities, output_path, title="Risk Distribution"):
    """Plot histogram of risk probabilities with category zones."""
    fig, ax = plt.subplots(figsize=(10, 6))

    probs_pct = np.array(probabilities) * 100

    ax.hist(probs_pct, bins=40, color='#e94560', edgecolor='white',
            linewidth=0.5, alpha=0.85)

    # Risk zones
    ax.axvspan(0, 30, alpha=0.1, color='#2ecc71', label='Low Risk')
    ax.axvspan(30, 60, alpha=0.1, color='#f39c12', label='Moderate Risk')
    ax.axvspan(60, 100, alpha=0.1, color='#e74c3c', label='High Risk')

    ax.set_xlabel('Risk Probability (%)', fontsize=13)
    ax.set_ylabel('Number of Patients', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.8)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved risk distribution: {output_path}")


def plot_feature_importance(importances, feature_names, output_path,
                            title="Feature Importance", top_n=15):
    """Plot feature importance bar chart."""
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.magma(np.linspace(0.3, 0.85, len(sorted_idx)))
    y_pos = range(len(sorted_idx))

    ax.barh(y_pos, importances[sorted_idx][::-1],
            color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]], fontsize=11)
    ax.set_xlabel('Importance', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved feature importance: {output_path}")


def generate_all_visualizations(eval_results, output_dir, disease_name=""):
    """
    Generate all visualization plots for a disease target.

    Parameters:
        eval_results: dict of {model_name: {metrics, y_true, y_pred, y_prob, model}}
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    disease_slug = disease_name.lower().replace(' ', '_')

    # ROC curves
    roc_data = {}
    for name, data in eval_results.items():
        if 'y_prob' in data and data['y_prob'] is not None:
            roc_data[name] = (data['y_true'], data['y_prob'])

    if roc_data:
        plot_roc_curves(
            roc_data,
            os.path.join(output_dir, f'roc_curves_{disease_slug}.png'),
            title=f'ROC Curves – {disease_name}'
        )

    # Confusion matrices
    for name, data in eval_results.items():
        plot_confusion_matrix(
            data['y_true'], data['y_pred'],
            os.path.join(output_dir, f'cm_{name.lower().replace(" ", "_")}_{disease_slug}.png'),
            title=f'{name} – {disease_name} Confusion Matrix'
        )

    # Model comparison
    comparison_data = {name: data['metrics'] for name, data in eval_results.items()}
    plot_model_comparison(
        comparison_data,
        os.path.join(output_dir, f'comparison_{disease_slug}.png'),
        disease_name=disease_name
    )

    # Risk distribution (using best model by AUC)
    best_model = max(eval_results.items(),
                     key=lambda x: x[1]['metrics'].get('roc_auc', 0))
    if best_model[1].get('y_prob') is not None:
        plot_risk_distribution(
            best_model[1]['y_prob'],
            os.path.join(output_dir, f'risk_dist_{disease_slug}.png'),
            title=f'{disease_name} – Risk Distribution ({best_model[0]})'
        )

    # Feature importance for tree models
    for name, data in eval_results.items():
        model = data.get('model')
        if model and hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model.feature_importances_,
                data['feature_names'],
                os.path.join(output_dir, f'feat_imp_{name.lower().replace(" ", "_")}_{disease_slug}.png'),
                title=f'{name} – {disease_name} Feature Importance'
            )
