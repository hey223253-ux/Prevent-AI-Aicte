# Fairness Report – Logistic Regression (Hypertension)

## Medical Disclaimer

> ⚠️ This system is for educational and research purposes only. It is NOT intended for clinical use. AI predictions may contain biases and should never replace professional medical judgment.


## Gender Fairness Analysis

| group   |   n_samples |   positive_rate |   predicted_positive_rate |   accuracy |   f1_score |   roc_auc |   true_positive_rate |   false_positive_rate |
|:--------|------------:|----------------:|--------------------------:|-----------:|-----------:|----------:|---------------------:|----------------------:|
| Female  |         461 |       0.0867679 |                  0.232104 |   0.828633 |   0.462585 |  0.917874 |             0.85     |              0.173397 |
| Male    |         539 |       0.0853432 |                  0.211503 |   0.836735 |   0.45     |  0.881118 |             0.782609 |              0.158215 |

**Equalized Odds**: Model shows approximate equalized odds across groups.

**Demographic Parity**: Model shows approximate demographic parity.


## Age Group Fairness Analysis

| group   |   n_samples |   positive_rate |   predicted_positive_rate |   accuracy |   f1_score |   roc_auc |   true_positive_rate |   false_positive_rate |
|:--------|------------:|----------------:|--------------------------:|-----------:|-----------:|----------:|---------------------:|----------------------:|
| 18-30   |          95 |       0.0947368 |                  0.168421 |   0.884211 |   0.56     |  0.900517 |             0.777778 |              0.104651 |
| 31-45   |         294 |       0.0986395 |                  0.217687 |   0.860544 |   0.55914  |  0.921535 |             0.896552 |              0.143396 |
| 46-60   |         389 |       0.0642674 |                  0.228792 |   0.804627 |   0.333333 |  0.863846 |             0.76     |              0.192308 |
| 61-75   |         174 |       0.12069   |                  0.264368 |   0.810345 |   0.507463 |  0.902583 |             0.809524 |              0.189542 |
| 75+     |          48 |       0.0416667 |                  0.125    |   0.875    |   0.25     |  0.891304 |             0.5      |              0.108696 |

**Equalized Odds**: Model may have fairness concerns: TPR range=0.397, FPR range=0.088 (threshold=0.1)

**Demographic Parity**: Model may have demographic parity concerns: prediction rate range=0.139


## Bias Detection Discussion

### Potential Sources of Bias

1. **Training Data Bias**: Synthetic data may not accurately represent the real-world distribution of disease risk across demographics. Real datasets (e.g., PIMA) are limited to specific populations.

2. **Feature Representation**: Certain features (e.g., family history) may correlate with socioeconomic factors not captured in the model.

3. **Label Bias**: Disease diagnoses in historical data may reflect disparities in healthcare access rather than true prevalence.

4. **Measurement Bias**: Wearable device data may vary in accuracy across different user demographics.

### Mitigation Strategies

- Regular fairness audits across demographic slices
- Adversarial debiasing during model training
- Calibration adjustments per demographic group
- Transparent reporting of model limitations
- Human-in-the-loop review for high-risk predictions
