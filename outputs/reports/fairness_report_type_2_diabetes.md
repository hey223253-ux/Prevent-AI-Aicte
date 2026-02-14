# Fairness Report – Logistic Regression (Type 2 Diabetes)

## Medical Disclaimer

> ⚠️ This system is for educational and research purposes only. It is NOT intended for clinical use. AI predictions may contain biases and should never replace professional medical judgment.


## Gender Fairness Analysis

| group   |   n_samples |   positive_rate |   predicted_positive_rate |   accuracy |   f1_score |   roc_auc |   true_positive_rate |   false_positive_rate |
|:--------|------------:|----------------:|--------------------------:|-----------:|-----------:|----------:|---------------------:|----------------------:|
| Female  |         461 |       0.0737527 |                  0.201735 |   0.83731  |   0.409449 |  0.921821 |             0.764706 |              0.156909 |
| Male    |         539 |       0.0723562 |                  0.222635 |   0.827458 |   0.415094 |  0.911846 |             0.846154 |              0.174    |

**Equalized Odds**: Model shows approximate equalized odds across groups.

**Demographic Parity**: Model shows approximate demographic parity.


## Age Group Fairness Analysis

| group   |   n_samples |   positive_rate |   predicted_positive_rate |   accuracy |   f1_score |   roc_auc |   true_positive_rate |   false_positive_rate |
|:--------|------------:|----------------:|--------------------------:|-----------:|-----------:|----------:|---------------------:|----------------------:|
| 18-30   |          95 |       0.0842105 |                  0.231579 |   0.810526 |   0.4      |  0.880747 |             0.75     |              0.183908 |
| 31-45   |         294 |       0.0884354 |                  0.238095 |   0.829932 |   0.479167 |  0.939294 |             0.884615 |              0.175373 |
| 46-60   |         389 |       0.0488432 |                  0.182519 |   0.845758 |   0.333333 |  0.920057 |             0.789474 |              0.151351 |
| 61-75   |         174 |       0.103448  |                  0.224138 |   0.833333 |   0.491228 |  0.909544 |             0.777778 |              0.160256 |
| 75+     |          48 |       0.0416667 |                  0.229167 |   0.770833 |   0.153846 |  0.793478 |             0.5      |              0.217391 |

**Equalized Odds**: Model may have fairness concerns: TPR range=0.385, FPR range=0.066 (threshold=0.1)

**Demographic Parity**: Model shows approximate demographic parity.


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
