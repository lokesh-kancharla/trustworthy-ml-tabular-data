from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_loader import load_or_download

RANDOM_STATE = 42
TARGET = "default"
RESULTS_DIR = Path("results")


def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def score(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def add_missingness(X, fraction=0.10, seed=42):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    numeric = Xc.select_dtypes(include="number").columns
    if len(numeric) == 0:
        return Xc
    mask = rng.random((len(Xc), len(numeric))) < fraction
    Xc.loc[:, numeric] = Xc.loc[:, numeric].mask(mask)
    return Xc


def add_numeric_noise(X, scale=0.10, seed=42):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    numeric = Xc.select_dtypes(include="number").columns
    for col in numeric:
        std = Xc[col].std()
        if pd.notna(std) and std > 0:
            Xc[col] = Xc[col] + rng.normal(0, std * scale, len(Xc))
    return Xc


def run():
    df = load_or_download()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    rows = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        conditions = {
            "clean": X_test,
            "10pct_missing": add_missingness(X_test, 0.10),
            "10pct_numeric_noise": add_numeric_noise(X_test, 0.10),
            "25pct_numeric_noise": add_numeric_noise(X_test, 0.25),
        }
        for condition, X_eval in conditions.items():
            metrics = score(pipeline, X_eval, y_test)
            rows.append({"model": name, "condition": condition, **metrics})

    result = pd.DataFrame(rows)
    clean_auc = result[result.condition == "clean"].set_index("model")["roc_auc"]
    result["roc_auc_drop_vs_clean"] = result.apply(
        lambda r: clean_auc[r["model"]] - r["roc_auc"], axis=1
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    result.to_csv(RESULTS_DIR / "robustness_metrics.csv", index=False)
    summary = {
        "dataset_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "models": list(models),
        "conditions": sorted(result.condition.unique().tolist()),
    }
    (RESULTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
