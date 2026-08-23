import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def inject_missingness(X: pd.DataFrame, fraction=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    out = X.copy()
    mask = rng.random(out.shape) < fraction
    return out.mask(mask)


def add_numeric_noise(X: pd.DataFrame, scale=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    out = X.copy()
    numeric = out.select_dtypes(include="number").columns
    for col in numeric:
        std = out[col].std()
        if pd.notna(std) and std > 0:
            out[col] = out[col] + rng.normal(0, std * scale, len(out))
    return out


def evaluate_shift(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def compare_robustness(model, X_test, y_test):
    scenarios = {
        "clean": X_test,
        "missing_05": inject_missingness(X_test, 0.05),
        "missing_15": inject_missingness(X_test, 0.15),
        "noise_05": add_numeric_noise(X_test, 0.05),
        "noise_15": add_numeric_noise(X_test, 0.15),
    }
    return {name: evaluate_shift(model, X, y_test) for name, X in scenarios.items()}
