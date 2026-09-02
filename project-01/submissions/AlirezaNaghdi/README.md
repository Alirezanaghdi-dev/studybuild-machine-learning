# Credit Risk Analysis and Prediction

An end-to-end machine-learning project for analyzing credit-card repayment behavior, predicting next-month payment default, assigning probability-based risk groups, and exploring customer segments with unsupervised learning.

The main supervised model is Logistic Regression. The project also includes SMOTENC for class imbalance, Optuna for hyperparameter optimization, SHAP and RFE for feature analysis, PCA/UMAP/t-SNE for visualization, UMAP + Gaussian Mixture Model (GMM) for behavioral clustering, and Random Forest for learning the cluster assignments.

> **Important:** Logistic Regression is the primary default-prediction model. UMAP, GMM, and the cluster-based Random Forest are supplementary analyses and do not replace the main credit-risk model.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Analytical Workflow](#analytical-workflow)
- [Data Preparation](#data-preparation)
- [Modeling Approach](#modeling-approach)
- [Supplementary Clustering](#supplementary-clustering)
- [Model Explainability](#model-explainability)
- [Main Results](#main-results)
- [Generated Files](#generated-files)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Reproducibility](#reproducibility)
- [Interpretation and Business Use](#interpretation-and-business-use)
- [Limitations](#limitations)
- [Author](#author)

## Project Objectives

This project addresses the following practical credit-risk questions:

1. What is the overall condition and quality of the data?
2. How do defaulting and non-defaulting customers differ?
3. Which repayment behaviors provide the strongest warning signals?
4. Does imbalance handling improve default detection?
5. Which evaluation metrics are suitable for an imbalanced target?
6. How do false negatives and false positives affect business decisions?
7. How can customers be divided into low-, medium-, and high-risk groups?
8. Which variables have the greatest influence on model predictions?
9. How sensitive are the results to the selected probability threshold?
10. What management actions can be taken for each risk group?

The project combines descriptive analysis, predictive modeling, model interpretation, threshold analysis, and behavioral segmentation in one workflow.

## Dataset

The project uses the **Default of Credit Card Clients** dataset.

- Observations: **30,000 customers**
- Raw columns: **25**
- Predictor variables used for modeling: **23**
- Target: `default payment next month`
- Non-default observations: **23,364**
- Default observations: **6,636**
- Default rate: **22.12%**
- Missing values: **0**
- Fully duplicated rows: **0**
- Duplicate customer IDs: **0**

### Variable Groups

| Group | Variables |
|---|---|
| Credit and demographic variables | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Repayment status | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| Monthly bill amounts | `BILL_AMT1` to `BILL_AMT6` |
| Monthly payment amounts | `PAY_AMT1` to `PAY_AMT6` |
| Target | `default payment next month` |

The dataset file must be stored at:

```text
dataset/default of credit card clients.xls
```

## Project Structure

```text
credit-risk-project/
├── cluster_dataset/
│   ├── clusters_train_test_final.xlsx
│   └── cluster_summary.xlsx
├── dataset/
│   └── default of credit card clients.xls
├── figures/
│   └── generated PNG figures
├── notebook/
│   └── Credit_Risk_Analysis_and_Prediction_Report_Notebook.ipynb
├── outputs/
│   ├── logistic_risk_summary_test.csv
│   ├── logistic_risk_thresholds.csv
│   ├── risk_groups_logistic.csv
│   └── risk_groups_rf_cluster.csv
├── report/
│   └── Credit Risk Analysis and Prediction Project Report.pdf
├── src/
│   └── model.py
├── terminal_output/
│   └── terminal_output.txt
└── README.md
```

The Python file uses paths relative to the `src` directory. Run the program from inside `src` so the dataset and output paths resolve correctly.

## Analytical Workflow

The complete workflow consists of the following stages:

1. Load the Excel dataset and inspect its shape, data types, missing values, duplicates, and IDs.
2. Clean and document categorical encodings.
3. Perform exploratory data analysis on demographic, credit, bill, payment, and repayment-status variables.
4. Compare defaulting and non-defaulting customers.
5. Engineer exploratory repayment and debt indicators.
6. Analyze the sensitivity of the engineered repayment-ratio threshold.
7. Perform PCA on payment amounts.
8. Split the original predictors into training and test data using stratification.
9. Apply variable-specific preprocessing.
10. Train a simple Logistic Regression baseline.
11. Train an optimized Logistic Regression pipeline with SMOTENC.
12. Compare fixed decision thresholds of 0.30, 0.50, and 0.70.
13. Evaluate ROC, Precision-Recall, cross-validation, and confusion-matrix results.
14. Explain important variables with coefficients, SHAP, and RFE.
15. Compare PCA, UMAP, and t-SNE on the complete transformed dataset.
16. Create three behavioral clusters with UMAP and GMM.
17. Name the clusters Low Risk, Medium Risk, and High Risk using training-set default rates.
18. Train an Optuna-tuned Random Forest to reproduce GMM cluster labels.
19. Export customer risk groups, cluster assignments, summaries, figures, and terminal output.

## Data Preparation

### Categorical Cleaning

- `EDUCATION` codes 5 and 6 are consolidated into code 0 (`Unknown`).
- Special `PAY_*` status codes are retained because they contain repayment information.
- Nominal variables are encoded separately from continuous and ordinal variables.

### Train/Test Split

The dataset is divided using a stratified split:

- Training observations: **24,000**
- Test observations: **6,000**
- Test size: **20%**
- Random state: **42**

Stratification preserves approximately the same default ratio in the training and test sets.

### Standard Preprocessing Pipeline

Continuous variables pass through the following steps:

1. **Winsorization:** caps the lower and upper 5% tails to reduce the influence of extreme values.
2. **Yeo-Johnson transformation:** reduces skewness and supports zero or negative values.
3. **RobustScaler:** scales values using statistics that are less sensitive to outliers.

The remaining variables are processed as follows:

- Ordinal repayment-status variables are passed through without one-hot encoding.
- Nominal variables (`SEX`, `EDUCATION`, and `MARRIAGE`) are one-hot encoded.

All preprocessing parameters are fitted on training data and then applied to test data. The transformed modeling data contains **31 features**.

### SMOTENC Pipeline

SMOTENC is used because the data contains both continuous and categorical predictors. The pipeline works in this order:

1. Transform continuous variables with winsorization, Yeo-Johnson, and RobustScaler.
2. Convert categorical variables to ordinal numeric codes.
3. Apply SMOTENC only to the training data or the training fold inside cross-validation.
4. One-hot encode the categorical columns after synthetic sampling.
5. Fit Logistic Regression on the resulting balanced training data.

This order allows SMOTENC to recognize categorical columns correctly and prevents synthetic fractional category values. The untouched test set is never oversampled.

## Modeling Approach

### 1. Simple Logistic Regression Baseline

The baseline model uses the standard transformed predictors without resampling. It provides an interpretable reference for evaluating whether imbalance handling improves the detection of defaulting customers.

### 2. Optimized Logistic Regression with SMOTENC

The optimized pipeline applies SMOTENC and tunes the Logistic Regression regularization parameter `C` with Optuna.

- Optimization method: Optuna TPE sampler
- Number of trials: **20**
- Validation method: **5-fold StratifiedKFold**
- Optimization metric: **ROC-AUC**
- Best `C`: **0.0114937776**
- Best cross-validated ROC-AUC during tuning: **0.7744**

### 3. Decision-Threshold Analysis

Both the baseline and optimized Logistic Regression models are evaluated at three fixed thresholds:

| Threshold | Interpretation | Main effect |
|---:|---|---|
| 0.30 | Aggressive detection | Higher recall and more false alerts |
| 0.50 | Standard decision | Balanced operational reference |
| 0.70 | Conservative review | Higher precision and more missed defaults |

These thresholds are comparison scenarios. The most suitable operational threshold depends on the financial cost of missed defaults and the available capacity for manual review.

### 4. Probability-Based Risk Groups

Risk-group boundaries are learned from validation probabilities produced without oversampling:

- Low/Medium boundary: **0.1667**
- Medium/High boundary: **0.3253**
- Low Risk: lower 50% of validation probabilities
- Medium Risk: between the 50th and 80th percentiles
- High Risk: upper 20% of validation probabilities

The validation set is used to define the boundaries so the test set remains untouched until final reporting.

## Supplementary Clustering

### PCA, UMAP, and t-SNE Comparison

PCA, UMAP, and t-SNE are applied to the complete preprocessed dataset and displayed side by side. Their role is visualization rather than supervised default prediction.

- **PCA** provides a linear two-dimensional projection.
- **UMAP** preserves useful local and broader nonlinear structure.
- **t-SNE** emphasizes local neighborhood separation.

These visualizations use all transformed observations. The target is used only for coloring the points and is not supplied as an input feature.

### UMAP + Gaussian Mixture Model

The clustering stage first reduces the transformed predictors to three UMAP dimensions and then applies a three-component GMM.

Final settings:

- UMAP neighbors: **44**
- UMAP minimum distance: **0.0027571**
- UMAP components: **3**
- GMM components: **3**
- GMM covariance type: `full`

GMM models each cluster as a multivariate Gaussian distribution in the three-dimensional UMAP representation. GMM supports probabilistic, potentially overlapping clusters rather than forcing only hard geometric boundaries.

The UMAP input has already passed through winsorization, Yeo-Johnson transformation, scaling, and categorical encoding. However, this does not mean the original variables become perfectly normal. The Gaussian assumption belongs to the GMM components in UMAP space.

### Assigning Risk Names to Clusters

GMM creates numeric labels without a risk meaning. Risk names are therefore learned using default rates from the **training set only**:

| Cluster | Assigned label | Train default rate | Test default rate |
|---:|---|---:|---:|
| 0 | Low Risk | 10.32% | 11.25% |
| 2 | Medium Risk | 16.06% | 16.72% |
| 1 | High Risk | 47.51% | 46.12% |

Silhouette scores:

- Training: **0.4926**
- Test: **0.4940**

The similar training and test results indicate reasonably stable cluster separation for this split.

### Random Forest for Cluster Labels

Random Forest is trained to reproduce the GMM cluster assignments from the original modeling predictors. It predicts **cluster membership**, not next-month default.

Optuna tunes the following parameters over 20 trials:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Best parameters from the recorded run:

```text
n_estimators     = 218
max_depth        = 15
min_samples_split = 6
min_samples_leaf  = 1
```

The Random Forest achieved **0.9845 test accuracy** for cluster-label prediction and **0.9892 mean five-fold cross-validation accuracy**.

The high score should not be interpreted as 98.45% default-prediction accuracy. It measures agreement with GMM cluster labels.

## Model Explainability

The project uses three complementary approaches:

### Logistic Regression Coefficients

Coefficients show the direction and relative strength of the relationship between transformed predictors and predicted default risk. The strongest positive coefficient in the recorded run is `PAY_0 = 2`, which represents a two-month recent delay.

### SHAP

SHAP measures how individual variables contribute to predictions.

Leading Logistic Regression SHAP variables include:

- `PAY_0`
- `EDUCATION`
- `BILL_AMT6`
- `PAY_AMT2`
- `LIMIT_BAL`

Leading Random Forest variables for High Risk cluster membership include:

- `PAY_3`
- `PAY_2`
- `PAY_0`
- `PAY_4`
- `PAY_6`

### Recursive Feature Elimination

RFE ranks all transformed features by repeatedly removing the least useful feature. The highest-ranked Logistic Regression features include recent payment-status categories, `LIMIT_BAL`, `PAY_AMT2`, and `BILL_AMT3`. The Random Forest RFE ranking is dominated by repayment-status variables.

SHAP, coefficients, and RFE describe predictive associations. They do not establish that a feature causes default.

## Main Results

### Default-Prediction Models

| Model | Accuracy | Default Precision | Default Recall | Default F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 0.8033 | 0.6450 | 0.2464 | 0.3566 | 0.7355 | 0.4971 |
| Logistic Regression + SMOTENC | 0.7493 | 0.4506 | 0.6081 | 0.5176 | 0.7590 | 0.5201 |

The SMOTENC model reduces overall accuracy but substantially improves recall and F1 for the default class. At the standard 0.50 threshold, missed defaults fall from **1,000** in the baseline model to **520** in the optimized model.

### Optimized Logistic Threshold Results

| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.2802 | 0.8862 | 0.4258 | 1,176 | 3,021 | 151 | 1,652 |
| 0.50 | 0.4506 | 0.6081 | 0.5176 | 807 | 984 | 520 | 3,689 |
| 0.70 | 0.6223 | 0.4084 | 0.4932 | 542 | 329 | 785 | 4,344 |

### Logistic Risk Groups on the Test Set

| Risk group | Customers | Mean predicted probability | Observed default rate |
|---|---:|---:|---:|
| Low | 2,920 | 10.40% | 11.71% |
| Medium | 1,895 | 22.38% | 18.42% |
| High | 1,185 | 50.79% | 53.67% |

The increasing observed default rate across the three groups shows that the probability-based segmentation provides a useful risk ranking.

## Generated Files

### Main CSV Outputs

| File | Description |
|---|---|
| `outputs/risk_groups_logistic.csv` | Predicted default probability and Logistic risk group for every customer |
| `outputs/logistic_risk_summary_test.csv` | Test-set counts, mean probabilities, and observed default rates by risk group |
| `outputs/logistic_risk_thresholds.csv` | Validation-based Low/Medium and Medium/High boundaries |
| `outputs/risk_groups_rf_cluster.csv` | Probability of High Risk cluster membership and supplementary cluster-based group |

### Clustering Outputs

| File | Description |
|---|---|
| `cluster_dataset/clusters_train_test_final.xlsx` | Original modeling data with cluster and risk labels in separate Train and Test sheets |
| `cluster_dataset/cluster_summary.xlsx` | Cluster size and summary statistics |

### Terminal Output

All printed results are captured in:

```text
terminal_output/terminal_output.txt
```

### Main Figures

The script saves figures at 300 DPI in the `figures` directory. They include:

- bill and payment distributions and boxplots;
- target and age-group distributions;
- repayment status by default class;
- credit-limit comparisons;
- numeric correlation matrix;
- repayment, bill, and payment separation;
- repayment-ratio sensitivity analysis;
- PCA payment biplot;
- Logistic confusion matrix, ROC curve, and Precision-Recall curve;
- Logistic coefficients, SHAP summary, and SHAP bar plot;
- PCA, UMAP, and t-SNE comparison;
- UMAP cluster and risk-label visualizations;
- Random Forest cluster confusion matrix;
- Random Forest High Risk ROC, Precision-Recall, and SHAP figures;
- observed default rate by Logistic risk group.

## Installation

### 1. Create a Virtual Environment

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib seaborn plotly scikit-learn imbalanced-learn feature-engine optuna umap-learn shap openpyxl xlrd python-docx PySide6
```

`PySide6` is required because the script explicitly selects Matplotlib's interactive `QtAgg` backend.

## How to Run

1. Place the dataset in the `dataset` directory with the exact filename:

```text
default of credit card clients.xls
```

2. Confirm that these directories exist before execution:

```text
cluster_dataset
dataset
figures
outputs
report
src
terminal_output
```

3. Open a terminal in the project root and run:

```bash
cd src
python model.py
```

The figures are displayed interactively and then saved. Closing each plot window allows the script to continue to the next stage.

The complete run may take time because it includes two Optuna studies, cross-validation, t-SNE, several UMAP transformations, GMM, SHAP, and RFE.

### Headless Environments

`QtAgg` requires a graphical desktop. When running on a server or another environment without a display, change:

```python
matplotlib.use("QtAgg")
```

to:

```python
matplotlib.use("Agg")
```

With `Agg`, figures are saved but cannot be displayed with `plt.show()`.

## Reproducibility

The project sets `RANDOM_STATE = 42` for:

- train/test splitting;
- StratifiedKFold;
- SMOTENC;
- Logistic Regression;
- Optuna TPE samplers;
- Random Forest;
- PCA, UMAP, t-SNE, and GMM;
- SHAP sampling.

These settings make repeated runs substantially more stable. Small differences may still occur across package versions, operating systems, numerical libraries, or hardware. Optuna may also select a nearby hyperparameter when the software environment or search history changes. Such small changes can slightly affect metrics and feature rankings without changing the main conclusions.

For strict reproducibility, save the exact dependency versions after a successful run:

```bash
python -m pip freeze > requirements.txt
```

## Interpretation and Business Use

The model should support risk prioritization rather than automatically approve or reject customers.

- **High-risk customers:** prioritize human review, early contact, closer payment monitoring, and suitable credit-control or repayment-support actions.
- **Medium-risk customers:** apply periodic monitoring and an early-warning process for changes in repayment behavior.
- **Low-risk customers:** use the standard process and consider appropriate benefits only after confirming repayment capacity and policy compliance.

The probability threshold should be selected according to the relative cost of false negatives and false positives. A lower threshold detects more potential defaults but increases operational review volume. A higher threshold reduces false alerts but misses more defaults.

Cluster labels can help tailor customer-management actions, but they must be treated as supplementary behavioral segments. The Logistic Regression probability remains the main output for default-risk ranking.

## Limitations

- The data represents a specific historical customer population and may not generalize directly to another institution, country, or period.
- The model requires temporal and external validation before operational use.
- SMOTENC improves minority-class detection but also increases false positives.
- Probability calibration should be checked before interpreting a score as an exact default probability.
- GMM assumptions apply in reduced UMAP space and do not prove that the original features are normally distributed.
- Cluster names depend on training-set default rates and must be revalidated when new data is introduced.
- Demographic variables require fairness testing and regulatory review.
- Feature importance describes prediction, not causation.
- Data drift, model performance, calibration, and fairness should be monitored after deployment.

## Author

**Alireza Naghdi**  
StudyBuild Credit Risk Analysis and Prediction Project, 2026
