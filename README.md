# Trustworthy Machine Learning for Tabular Data

Research-oriented machine learning project examining how predictive models behave when real-world tabular data are imbalanced, noisy, incomplete, or shifted.

## Research Question
**How robust are common machine-learning models for credit-default prediction under class imbalance, missing values, feature noise, and distribution shift?**

## Planned Study
- Perform exploratory analysis and data-quality assessment on a public credit-default dataset.
- Build reproducible preprocessing and feature-engineering pipelines.
- Establish baseline classifiers such as Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Compare Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrices.
- Stress-test models by introducing controlled missingness, noise, imbalance, and distribution shifts.
- Examine feature importance and model interpretability.
- Document failure modes, limitations, and reliability trade-offs rather than reporting accuracy alone.

## Research Themes
`Trustworthy AI` `Tabular ML` `Robustness` `Class Imbalance` `Explainability` `Model Evaluation`

## Technology
Python • Pandas • NumPy • scikit-learn • Matplotlib • Jupyter

## Repository Structure
```text
notebooks/   exploratory analysis and experiments
src/         reusable preprocessing, training, and evaluation code
data/        dataset instructions (raw data not committed when restricted)
results/     figures, tables, and experiment summaries
docs/        methodology and research notes
```

## Status
🧪 **In development.** Experiments, code, results, and methodology will be added incrementally. The objective is to develop this repository as a reproducible research-style study rather than a single tutorial notebook.

## Author
**Lokesh Kancharla** — M.S. Computer Science, University of Memphis
