# Trustworthy Machine Learning for Tabular Data

A research-oriented machine-learning project examining whether common tabular models remain reliable when real-world data quality changes.

## Research Question
**How robust are credit-default prediction models under class imbalance, missing values, and numerical feature noise?**

## Why this project
A model can perform well on a clean test set and still fail when deployed on imperfect data. This project therefore evaluates both **predictive performance** and **performance degradation under controlled perturbations**.

## Data Source
**UCI Machine Learning Repository — Default of Credit Card Clients (dataset id 350)**

The loader in `src/data_loader.py` retrieves the dataset using the `ucimlrepo` Python package, combines features and the binary target, renames the target to `default`, and caches it locally as `data/credit_default.csv`.

The raw data are not manually fabricated for this project. The loader makes the acquisition step explicit and reproducible.

## Models
- Logistic Regression — linear/interpretable baseline
- Random Forest — nonlinear tree ensemble
- Gradient Boosting — boosted tree ensemble for tabular prediction

## Preprocessing
Numeric features:
- median imputation
- standard scaling

Categorical features:
- most-frequent imputation
- one-hot encoding

All preprocessing is wrapped inside scikit-learn pipelines to keep training/evaluation transformations consistent and reduce leakage risk.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

Because default prediction is imbalanced, accuracy alone is not treated as sufficient.

## Robustness Experiments
`src/experiment.py` evaluates each fitted model under:
1. clean test data
2. 10% random numerical missingness
3. 10% numerical noise relative to each feature's standard deviation
4. 25% numerical noise

The experiment also calculates the **ROC-AUC drop relative to clean performance**. Smaller degradation suggests better stability under that perturbation.

## Repository Structure
```text
src/
  data_loader.py     real UCI dataset acquisition
  train.py           baseline model comparison
  robustness.py      reusable perturbation helpers
  experiment.py      end-to-end robustness experiment

data/                locally cached dataset (ignored when appropriate)
results/             generated experiment metrics

docs/
  EXPLAIN_ME.md      professor/interview walkthrough
requirements.txt
```

## Run the Project
```bash
pip install -r requirements.txt
python src/experiment.py
```

Generated outputs are written to `results/`, including `robustness_metrics.csv` and an experiment summary.

## Research Extensions
- probability calibration
- subgroup/fairness analysis
- temporal or synthetic distribution shift
- SHAP-based explanation
- repeated cross-validation and confidence intervals
- XGBoost/LightGBM or neural tabular models

## Limitations
- Controlled missingness/noise does not represent every deployment shift.
- This project does not claim to establish safety or fairness.
- A historical credit dataset may not represent current populations or lending environments.
- Results should be interpreted together with calibration, subgroup behavior, and application-specific costs.

## Research Themes
`Trustworthy AI` `Tabular ML` `Robustness` `Class Imbalance` `Explainability` `Model Evaluation`

## Author
**Lokesh Kancharla** — M.S. Computer Science, University of Memphis
