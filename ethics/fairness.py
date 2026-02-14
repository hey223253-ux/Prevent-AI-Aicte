"""
PreventAI – Fairness & Ethics Module
=======================================
Evaluates model fairness across demographic groups (age, gender).
Includes bias detection and equalized odds analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def define_age_groups(ages):
    """Bin ages into demographic groups."""
    bins = [0, 30, 45, 60, 75, 100]
    labels = ['18-30', '31-45', '46-60', '61-75', '75+']
    return pd.cut(ages, bins=bins, labels=labels, include_lowest=True)


def compute_group_metrics(y_true, y_pred, y_prob, groups, group_name="Group"):
    """
    Compute metrics for each demographic subgroup.

    Returns DataFrame with per-group metrics.
    """
    results = []
    unique_groups = sorted(groups.unique())

    for group in unique_groups:
        mask = groups == group
        n = mask.sum()
        if n < 10:
            continue

        metrics = {
            'group': group,
            'n_samples': n,
            'positive_rate': y_true[mask].mean(),
            'predicted_positive_rate': y_pred[mask].mean(),
            'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'f1_score': f1_score(y_true[mask], y_pred[mask], zero_division=0),
        }

        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true[mask], y_prob[mask])
            except ValueError:
                metrics['roc_auc'] = np.nan

        # True positive rate (sensitivity / recall)
        tp = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        fn = ((y_pred[mask] == 0) & (y_true[mask] == 1)).sum()
        metrics['true_positive_rate'] = tp / max(tp + fn, 1)

        # False positive rate
        fp = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum()
        tn = ((y_pred[mask] == 0) & (y_true[mask] == 0)).sum()
        metrics['false_positive_rate'] = fp / max(fp + tn, 1)

        results.append(metrics)

    return pd.DataFrame(results)


def check_equalized_odds(group_metrics, threshold=0.1):
    """
    Check if equalized odds condition is approximately satisfied.
    Equalized odds requires similar TPR and FPR across groups.
    """
    tpr_range = group_metrics['true_positive_rate'].max() - group_metrics['true_positive_rate'].min()
    fpr_range = group_metrics['false_positive_rate'].max() - group_metrics['false_positive_rate'].min()

    equalized = tpr_range < threshold and fpr_range < threshold

    return {
        'equalized_odds_satisfied': equalized,
        'tpr_range': tpr_range,
        'fpr_range': fpr_range,
        'threshold': threshold,
        'interpretation': (
            'Model shows approximate equalized odds across groups.'
            if equalized else
            f'Model may have fairness concerns: TPR range={tpr_range:.3f}, '
            f'FPR range={fpr_range:.3f} (threshold={threshold})'
        )
    }


def check_demographic_parity(group_metrics, threshold=0.1):
    """
    Check demographic parity: similar prediction rates across groups.
    """
    pred_rate_range = (group_metrics['predicted_positive_rate'].max() -
                       group_metrics['predicted_positive_rate'].min())

    satisfied = pred_rate_range < threshold

    return {
        'demographic_parity_satisfied': satisfied,
        'prediction_rate_range': pred_rate_range,
        'threshold': threshold,
        'interpretation': (
            'Model shows approximate demographic parity.'
            if satisfied else
            f'Model may have demographic parity concerns: '
            f'prediction rate range={pred_rate_range:.3f}'
        )
    }


def plot_fairness_analysis(group_metrics, group_name, output_path, disease_name=""):
    """Generate fairness visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    groups = group_metrics['group'].astype(str).values
    x = np.arange(len(groups))

    # Accuracy by group
    axes[0].bar(x, group_metrics['accuracy'], color='#e94560',
                edgecolor='white', linewidth=0.5, alpha=0.9)
    axes[0].set_title(f'Accuracy by {group_name}', fontsize=13, fontweight='bold', color='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)

    # TPR and FPR by group
    width = 0.35
    axes[1].bar(x - width/2, group_metrics['true_positive_rate'],
                width, label='TPR', color='#48c9b0', edgecolor='white')
    axes[1].bar(x + width/2, group_metrics['false_positive_rate'],
                width, label='FPR', color='#e74c3c', edgecolor='white')
    axes[1].set_title(f'TPR & FPR by {group_name}', fontsize=13, fontweight='bold', color='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=45, ha='right')
    axes[1].legend(framealpha=0.8)
    axes[1].set_ylim(0, 1)

    # Positive prediction rate (demographic parity)
    axes[2].bar(x, group_metrics['predicted_positive_rate'],
                color='#f39c12', edgecolor='white', linewidth=0.5, alpha=0.9)
    axes[2].set_title(f'Prediction Rate by {group_name}', fontsize=13, fontweight='bold', color='white')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(groups, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#ccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle(f'Fairness Analysis – {disease_name}',
                 fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved fairness plot: {output_path}")


def generate_fairness_report(
    y_true, y_pred, y_prob, patient_data,
    model_name, disease_name, output_dir
):
    """
    Full fairness evaluation pipeline.

    Parameters:
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities
        patient_data: DataFrame with 'age' and 'gender' columns
        model_name: name of the model
        disease_name: name of the disease
        output_dir: directory for outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    disease_slug = disease_name.lower().replace(' ', '_')

    report_lines = []
    report_lines.append(f"# Fairness Report – {model_name} ({disease_name})\n")
    report_lines.append("## Medical Disclaimer\n")
    report_lines.append(
        "> ⚠️ This system is for educational and research purposes only. "
        "It is NOT intended for clinical use. AI predictions may contain biases "
        "and should never replace professional medical judgment.\n"
    )

    # Gender analysis
    if 'gender' in patient_data.columns:
        gender_groups = patient_data['gender'].values[:len(y_true)]
        gender_metrics = compute_group_metrics(
            y_true, y_pred, y_prob,
            pd.Series(gender_groups), "Gender"
        )

        report_lines.append("\n## Gender Fairness Analysis\n")
        report_lines.append(gender_metrics.to_markdown(index=False))

        eo = check_equalized_odds(gender_metrics)
        dp = check_demographic_parity(gender_metrics)

        report_lines.append(f"\n**Equalized Odds**: {eo['interpretation']}")
        report_lines.append(f"\n**Demographic Parity**: {dp['interpretation']}\n")

        plot_fairness_analysis(
            gender_metrics, "Gender",
            os.path.join(output_dir, f'fairness_gender_{disease_slug}.png'),
            disease_name
        )

    # Age group analysis
    if 'age' in patient_data.columns:
        age_groups = define_age_groups(patient_data['age'].values[:len(y_true)])
        age_metrics = compute_group_metrics(
            y_true, y_pred, y_prob,
            age_groups, "Age Group"
        )

        report_lines.append("\n## Age Group Fairness Analysis\n")
        report_lines.append(age_metrics.to_markdown(index=False))

        eo = check_equalized_odds(age_metrics)
        dp = check_demographic_parity(age_metrics)

        report_lines.append(f"\n**Equalized Odds**: {eo['interpretation']}")
        report_lines.append(f"\n**Demographic Parity**: {dp['interpretation']}\n")

        plot_fairness_analysis(
            age_metrics, "Age Group",
            os.path.join(output_dir, f'fairness_age_{disease_slug}.png'),
            disease_name
        )

    # Bias discussion
    report_lines.append("\n## Bias Detection Discussion\n")
    report_lines.append(
        "### Potential Sources of Bias\n\n"
        "1. **Training Data Bias**: Synthetic data may not accurately represent "
        "the real-world distribution of disease risk across demographics. "
        "Real datasets (e.g., PIMA) are limited to specific populations.\n\n"
        "2. **Feature Representation**: Certain features (e.g., family history) "
        "may correlate with socioeconomic factors not captured in the model.\n\n"
        "3. **Label Bias**: Disease diagnoses in historical data may reflect "
        "disparities in healthcare access rather than true prevalence.\n\n"
        "4. **Measurement Bias**: Wearable device data may vary in accuracy "
        "across different user demographics.\n\n"
        "### Mitigation Strategies\n\n"
        "- Regular fairness audits across demographic slices\n"
        "- Adversarial debiasing during model training\n"
        "- Calibration adjustments per demographic group\n"
        "- Transparent reporting of model limitations\n"
        "- Human-in-the-loop review for high-risk predictions\n"
    )

    # Save report
    report = "\n".join(report_lines)
    report_path = os.path.join(output_dir, f'fairness_report_{disease_slug}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Saved fairness report: {report_path}")

    return report
