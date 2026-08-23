# How to Explain This Project

## One-sentence summary
I built a tabular machine-learning study to test not only predictive performance, but also how model reliability changes when the data contain missing values and numerical noise.

## Why I built it
Most portfolio projects stop after reporting accuracy. I wanted to study a more research-oriented question: **does a model that performs well on clean data remain reliable when the data quality changes?** This connects to trustworthy AI because real-world systems often face missing values, noisy measurements, class imbalance, and distribution changes.

## Dataset
The project uses the UCI **Default of Credit Card Clients** dataset (UCI repository dataset id 350). `src/data_loader.py` downloads the dataset with `ucimlrepo`, combines features and target, renames the target to `default`, and caches the CSV locally.

## Preprocessing
`src/experiment.py` automatically separates numeric and categorical columns.

Numeric features:
- missing values → median imputation
- scaling → StandardScaler

Categorical features:
- missing values → most frequent category
- encoding → OneHotEncoder

The preprocessing is inside a scikit-learn Pipeline so training and evaluation use the same transformation logic and avoid accidental data leakage.

## Models
I compare three model families:
1. Logistic Regression — interpretable linear baseline.
2. Random Forest — nonlinear ensemble with feature interactions.
3. Gradient Boosting — sequential tree ensemble that often performs strongly on tabular data.

Class weighting is used where supported to make the baseline more sensitive to the minority/default class.

## Metrics
I report:
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

For an imbalanced risk problem, accuracy alone is not enough. Recall measures how many actual defaults are detected, precision measures how many predicted defaults are correct, and ROC-AUC measures ranking ability across thresholds.

## Robustness tests
After fitting each model on the same training split, I evaluate it under four conditions:
- clean test data
- 10% random missingness in numerical features
- 10% numerical noise relative to feature standard deviation
- 25% numerical noise

The project then calculates the drop in ROC-AUC relative to clean performance. A smaller drop indicates better stability under that perturbation.

## What I would say if asked about limitations
- The perturbations are controlled simulations, not every possible real-world shift.
- The study does not prove fairness or safety.
- ROC-AUC is only one reliability measure.
- The dataset is historical and application-specific.
- More rigorous future work could include temporal shift, subgroup analysis, calibration, SHAP explanations, repeated cross-validation, and statistical significance testing.

## Files to know
- `src/data_loader.py` — gets and caches the real UCI dataset.
- `src/train.py` — original baseline model comparison.
- `src/robustness.py` — reusable perturbation helpers.
- `src/experiment.py` — end-to-end robustness experiment.
- `results/` — generated metrics after running the experiment.

## How to run
```bash
pip install -r requirements.txt
python src/experiment.py
```

## Research extension ideas
If a professor asks what I would do next, I would propose:
1. probability calibration and Expected Calibration Error
2. temporal or synthetic distribution shift
3. subgroup robustness and fairness analysis
4. SHAP-based feature explanations
5. repeated cross-validation with confidence intervals
6. comparison with XGBoost / LightGBM / neural tabular models
