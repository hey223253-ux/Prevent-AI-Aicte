"""
PreventAI – Baseline Models
==============================
Logistic Regression, Random Forest, and XGBoost classifiers
for disease risk prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, y_train):
    """Logistic Regression with class balancing."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42,
        C=1.0
    )
    model.fit(X_train, y_train)
    print("  ✓ Logistic Regression trained")
    return model


def train_random_forest(X_train, y_train, tune=True):
    """Random Forest with optional hyperparameter tuning."""
    if tune:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [8, 12, 16],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
        }
        rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"  ✓ Random Forest trained (best params: {grid_search.best_params_})")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("  ✓ Random Forest trained")
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """XGBoost with early stopping."""
    # Calculate scale_pos_weight for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        print("  ✓ XGBoost trained with early stopping")
    else:
        model.fit(X_train, y_train)
        print("  ✓ XGBoost trained")

    return model


def cross_validate_model(model, X, y, cv=5, scoring='roc_auc'):
    """Perform stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    print(f"  ✓ {cv}-Fold CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def train_all_baseline_models(X_train, y_train, X_val=None, y_val=None, tune_rf=True):
    """Train all baseline models and return them in a dict."""
    print("\n▶ Training Baseline Models...\n")

    models = {}

    # Logistic Regression
    print("  [1/3] Logistic Regression")
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)

    # Random Forest
    print("  [2/3] Random Forest")
    models['Random Forest'] = train_random_forest(X_train, y_train, tune=tune_rf)

    # XGBoost
    print("  [3/3] XGBoost")
    models['XGBoost'] = train_xgboost(X_train, y_train, X_val, y_val)

    return models


def predict_with_model(model, X):
    """Get predictions and probabilities from a sklearn-compatible model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob
