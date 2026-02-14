# Fairness Report – Logistic Regression (Cardiovascular Disease)

## Medical Disclaimer

> ⚠️ This system is for educational and research purposes only. It is NOT intended for clinical use. AI predictions may contain biases and should never replace professional medical judgment.


## Gender Fairness Analysis

| group   |   n_samples |   positive_rate |   predicted_positive_rate |   accuracy |   f1_score |   roc_auc |   true_positive_rate |   false_positive_rate |
|:--------|------------:|----------------:|--------------------------:|-----------:|-----------:|----------:|---------------------:|----------------------:|
| Female  |         461 |       0.0542299 |                  0.190889 |   0.83731  |   0.336283 |  0.90422  |                 0.76 |              0.158257 |
| Male    |         539 |       0.0371058 |                  0.204082 |   0.825603 |   0.276923 |  0.938536 |                 0.9  |              0.177264 |

**Equalized Odds**: Model may have fairness concerns: TPR range=0.140, FPR range=0.019 (threshold=0.1)

**Demographic Parity**: Model shows approximate demographic parity.


## Age Group Fairness Analysis

| group   |   n_samples |   positive_rate |   predicted_positive_rate |   accuracy |   f1_score |   roc_auc |   true_positive_rate |   false_positive_rate |
|:--------|------------:|----------------:|--------------------------:|-----------:|-----------:|----------:|---------------------:|----------------------:|
| 18-30   |          95 |       0.105263  |                  0.273684 |   0.768421 |   0.388889 |  0.88     |                0.7   |             0.223529  |
| 31-45   |         294 |       0.0442177 |                  0.217687 |   0.826531 |   0.337662 |  0.951547 |                1     |             0.181495  |
| 46-60   |         389 |       0.0308483 |                  0.172237 |   0.858612 |   0.303797 |  0.970601 |                1     |             0.145889  |
| 61-75   |         174 |       0.045977  |                  0.201149 |   0.787356 |   0.139535 |  0.784639 |                0.375 |             0.192771  |
| 75+     |          48 |       0.0416667 |                  0.125    |   0.916667 |   0.5      |  1        |                1     |             0.0869565 |

**Equalized Odds**: Model may have fairness concerns: TPR range=0.625, FPR range=0.137 (threshold=0.1)

**Demographic Parity**: Model may have demographic parity concerns: prediction rate range=0.149


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
