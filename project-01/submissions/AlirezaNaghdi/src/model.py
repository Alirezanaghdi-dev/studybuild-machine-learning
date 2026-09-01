import os


# Disable TensorFlow oneDNN optimization.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
print("MATPLOTLIB BACKEND:", matplotlib.get_backend())

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import shap
import optuna
import umap
from scipy import stats
from feature_engine.outliers import Winsorizer
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, silhouette_score,)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler
from docx import Document
import builtins

all_outputs = []
original_print = print

def print(*args, **kwargs):
    text = " ".join(map(str, args))
    all_outputs.append(text)
    original_print(*args, **kwargs)

#****** Global settings ******

sns.set_style("whitegrid")

DATA_PATH = "../dataset/default of credit card clients.xls"
TARGET_COL = "default payment next month"
RANDOM_STATE = 42

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAYMENT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


#****** Helper functions ******

def distribution_plot(nrows, ncols, data, columns, save_name):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5))

    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"../figures/{save_name}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def boxplot_grid(nrows, ncols, data, columns, save_name):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5))

    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"../figures/{save_name}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def test_normality(data: pd.DataFrame, columns: list) -> None:
    """Run the Shapiro-Wilk normality test on each column."""
    print("Running normality test:")
    for column in columns:
        values = data[column].dropna()
        if len(values) > 5000:
            values = values.sample(n=5000, random_state=RANDOM_STATE)
        _, p_value = stats.shapiro(values)
        result = "Fail to reject normality." if p_value > 0.05 else "Reject normality."
        print(f"{column}: Shapiro-Wilk test p-value = {p_value:.4f}. {result}")

def apply_threshold(probabilities, threshold=0.5):
    return (probabilities >= threshold).astype(int)


#****** 1. Load data and inspect data quality ******

df = pd.read_excel(DATA_PATH, header=1)

print("\n=== Raw Dataset Shape ===")
print(df.shape)

print("\n=== Raw Missing Values ===")
print(df.isna().sum())

print("\n=== Raw Duplicate Rows ===")
print(df.duplicated().sum())

id_col = df.columns[0]
print(f"\nID column name: {id_col}")

print("\n=== Raw Duplicate IDs ===")
print(df[id_col].duplicated().sum())

print("\n=== Data Types ===")
print(df.dtypes)

# Check duplicates before cleaning.
dups_full = df.duplicated()
dups_id = df.duplicated(subset=[id_col])

if dups_full.any():
    print("\nExample of fully duplicated rows:")
    print(df.loc[dups_full].head().to_string())

if dups_id.any():
    print("\nExample of duplicate IDs:")
    print(df.loc[dups_id].head().to_string())

# Check duplicates with ID as index.
df_indexed = df.set_index(id_col)
print("\n=== Duplicate Check with ID as Index ===")
print(f"Duplicated rows excluding ID: {df_indexed.duplicated().sum()}")
print(f"Duplicated index values: {df_indexed.index.duplicated().sum()}")

# Remove missing values and duplicates.
df_clean = df.dropna().drop_duplicates().drop_duplicates(subset=[id_col]).copy()

print("\n=== Clean Dataset Shape ===")
print(df_clean.shape)
print(df_clean.info())


#****** 2. Document and clean categorical encodings ******

sex_map = {1: "Male", 2: "Female", 3: "Others",}

pay_status_map = {-2: "Undocumented (-2)", -1: "Pay Duly", 0: "Undocumented (0)", 1: "1-Month Delay", 2: "2-Month Delay", 3: "3-Month Delay", 4: "4-Month Delay", 5: "5-Month Delay", 6: "6-Month Delay", 7: "7-Month Delay", 8: "8-Month Delay", 9: "9+ Months Delay",}

education_map = {0: "Unknown", 1: "Graduate School", 2: "University", 3: "High School", 4: "Others",}

marriage_map = {0: "Unknown", 1: "Married", 2: "Single", 3: "Others",}

target_map = {0: "Non-default", 1: "Default",}

print("\n=== SEX Values ===")
sex_counts = df_clean["SEX"].value_counts(dropna=False).sort_index()
sex_df = pd.DataFrame({"SEX": sex_counts.index, "SEX_name": sex_counts.index.map(sex_map), "count": sex_counts.values,})
print(sex_df.to_string(index=False))

print("\n=== EDUCATION Values Before Cleaning ===")
print(df_clean["EDUCATION"].value_counts(dropna=False).sort_index())

# Group EDUCATION values 5 and 6 as Unknown.
df_clean["EDUCATION"] = df_clean["EDUCATION"].replace([5, 6], 0)

print("\n=== EDUCATION Values After Cleaning ===")
education_counts = df_clean["EDUCATION"].value_counts(dropna=False).sort_index()
education_df = pd.DataFrame({"EDUCATION": education_counts.index, "education_name": education_counts.index.map(education_map), "count": education_counts.values,})
print(education_df.to_string(index=False))

print("\n=== MARRIAGE Values ===")
marriage_counts = df_clean["MARRIAGE"].value_counts(dropna=False).sort_index()
marriage_df = pd.DataFrame({"MARRIAGE": marriage_counts.index, "marriage_name": marriage_counts.index.map(marriage_map), "count": marriage_counts.values,})
print(marriage_df.to_string(index=False))

print("\n=== Target Values ===")
target_counts = df_clean[TARGET_COL].value_counts(dropna=False).sort_index()
target_df = pd.DataFrame({"TARGET": target_counts.index, "target_name": target_counts.index.map(target_map), "count": target_counts.values,})
print(target_df.to_string(index=False))

print("\n=== Repayment Status Encodings ===")
for code, label in pay_status_map.items():
    print(f"{code}: {label}")

print("\n=== AGE Range ===")
print(df_clean["AGE"].min(), df_clean["AGE"].max())

print("\n=== LIMIT_BAL Range ===")
print(df_clean["LIMIT_BAL"].min(), df_clean["LIMIT_BAL"].max())


#****** 3. Prepare an analysis copy for EDA ******

df_clean["age_group"] = pd.qcut(df_clean["AGE"], q=5, duplicates="drop")

df_analysis = df_clean.copy()
df_analysis["EDUCATION"] = df_analysis["EDUCATION"].map(education_map)
df_analysis["SEX"] = df_analysis["SEX"].map(sex_map)
df_analysis["MARRIAGE"] = df_analysis["MARRIAGE"].map(marriage_map)
df_analysis["default_label"] = df_analysis[TARGET_COL].map(target_map)

categorical_columns = ["SEX", "EDUCATION", "MARRIAGE", *PAY_COLS, "default_label",]

print("\n=== Unique Values in Categorical Columns ===")
for col in categorical_columns:
    print(f"{col}: {df_analysis[col].unique()}")

print("\n=== Analysis DataFrame Info ===")
print(df_analysis.info())

numeric_summary_source = df_clean.select_dtypes(include=np.number)
summary_describe = numeric_summary_source.describe().T
summary_describe["Skewness"] = numeric_summary_source.skew()
summary_describe["Kurtosis"] = numeric_summary_source.kurt()

print("\n=== Comprehensive Statistical Overview ===")
print(summary_describe.to_string())


#****** 4. Basic portfolio visualizations ******

distribution_plot(2, 3, df_analysis, BILL_COLS, "bill_amount_distributions.png")
distribution_plot(2, 3, df_analysis, PAYMENT_COLS, "payment_amount_distributions.png")
boxplot_grid(2, 3, df_analysis, BILL_COLS, "bill_amount_boxplots.png")
boxplot_grid(2, 3, df_analysis, PAYMENT_COLS, "payment_amount_boxplots.png")

plt.figure(figsize=(8, 5))
sns.countplot(data=df_analysis, x="default_label", order=["Non-default", "Default"])
plt.title("Target Class Distribution")
plt.xlabel("Default Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../figures/target_class_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()

print("\n=== Customers by Age Group ===")
print(df_analysis["age_group"].value_counts().sort_index())

plt.figure(figsize=(10, 6))
sns.countplot(data=df_analysis, x="age_group")
plt.title("Customers by Age Group", fontsize=16)
plt.xlabel("Age Group", fontsize=13)
plt.ylabel("Number of Customers", fontsize=13)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/customers_by_age_group.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.close()


#****** 5. Late-payment behavior by customer characteristics ******

df_analysis["late_payment_count"] = (df_analysis[PAY_COLS] > 0).sum(axis=1)
df_analysis["had_late_payment"] = (df_analysis["late_payment_count"] > 0).astype(int)

customer_group_columns = ["SEX", "MARRIAGE", "age_group", "EDUCATION"]

for col in customer_group_columns:
    group_summary = (df_analysis.groupby(col, as_index=False, observed=True).agg(customer_count=("ID", "count"), mean_limit_bal=("LIMIT_BAL", "mean"), median_limit_bal=("LIMIT_BAL", "median"), late_customer_count=("had_late_payment", "sum"), late_payment_rate=("had_late_payment", "mean"), avg_late_payment_count=("late_payment_count", "mean"),).query("customer_count >= 20").sort_values("mean_limit_bal"))

    print(f"\n=== Lowest Mean Credit Limits by {col} ===")
    print(group_summary.head(5).to_string(index=False))
    print(f"\n=== Highest Mean Credit Limits by {col} ===")
    print(group_summary.tail(5).to_string(index=False))

combined_group_summary = (df_analysis.groupby(customer_group_columns, as_index=False, observed=True).agg(customer_count=("ID", "count"), mean_limit_bal=("LIMIT_BAL", "mean"), median_limit_bal=("LIMIT_BAL", "median"), late_customer_count=("had_late_payment", "sum"), late_payment_rate=("had_late_payment", "mean"), avg_late_payment_count=("late_payment_count", "mean"),).query("customer_count >= 20").sort_values("mean_limit_bal"))

print("\n=== Combined Customer Group Summary: Lowest Mean Credit Limits ===")
print(combined_group_summary.head(10).to_string(index=False))

print("\n=== Combined Customer Group Summary: Highest Mean Credit Limits ===")
print(combined_group_summary.tail(10).to_string(index=False))

sex_marriage_summary = (df_analysis.groupby(["MARRIAGE", "SEX"], as_index=False, observed=True).agg(customer_count=("ID", "count"), mean_limit_bal=("LIMIT_BAL", "mean"), median_limit_bal=("LIMIT_BAL", "median"), late_customer_count=("had_late_payment", "sum"), late_payment_rate=("had_late_payment", "mean"), avg_late_payment_count=("late_payment_count", "mean"),).query("customer_count >= 20").sort_values("mean_limit_bal"))

print("\n=== Marital Status and Sex Summary ===")
print(sex_marriage_summary.to_string(index=False))

fig = px.bar(sex_marriage_summary, x="MARRIAGE", y="late_payment_rate", color="SEX", barmode="group", text_auto=True, title="Late Payment Rate by Marital Status and Sex",)
fig.write_html("../figures/late_payment_rate_by_marital_status_and_sex.html")
fig.show()


#****** 6. Default vs non-default comparisons ******

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, col in zip(axes, PAY_COLS):
    sns.countplot(data=df_analysis, x=col, hue="default_label", ax=ax)
    ax.set_title(f"{col} by Default Status")
    ax.set_xlabel("Repayment Status")
    ax.set_ylabel("Customer Count")

plt.tight_layout()
plt.savefig("../figures/repayment_status_by_default.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

for col in PAY_COLS:
    print(f"\n=== {col}: Distribution within Default Groups (%) ===")
    pay_table = pd.crosstab(df_analysis[col], df_analysis["default_label"], normalize="columns",) * 100
    print(pay_table.round(2))

late_summary = (df_analysis.groupby("default_label", as_index=False, observed=True).agg(late_payment_rate=("had_late_payment", "mean"), avg_late_payment_count=("late_payment_count", "mean"),))

print("\n=== Late Payment Summary by Default Status ===")
print(late_summary.to_string(index=False))

bill_summary = df_analysis.groupby("default_label", observed=True)[BILL_COLS].agg(["mean", "median"])
print("\n=== Bill Amount Summary by Default Status ===")
print(bill_summary.to_string())

payment_summary = df_analysis.groupby("default_label", observed=True)[PAYMENT_COLS].agg(["mean", "median"])
print("\n=== Payment Amount Summary by Default Status ===")
print(payment_summary.to_string())

limit_summary = (df_analysis.groupby("default_label", observed=True)["LIMIT_BAL"].agg(["count", "mean", "median"]).reset_index().sort_values("mean", ascending=False))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = ["count", "mean", "median"]
titles = ["Customer Count", "Mean Credit Limit", "Median Credit Limit"]
y_labels = ["Customers", "Mean LIMIT_BAL", "Median LIMIT_BAL"]

for ax, metric, title, ylabel in zip(axes, metrics, titles, y_labels):
    sns.barplot(data=limit_summary, x="default_label", y=metric, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Default Status")
    ax.set_ylabel(ylabel)

plt.tight_layout()
plt.savefig("../figures/credit_limit_summary_by_default.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_analysis, x="default_label", y="LIMIT_BAL")
plt.title("Credit Limit by Default Status")
plt.xlabel("Default Status")
plt.ylabel("LIMIT_BAL")
plt.tight_layout()
plt.savefig("../figures/credit_limit_boxplot_by_default.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

numeric_features = ["LIMIT_BAL", "AGE", *PAY_COLS, *BILL_COLS, *PAYMENT_COLS,]

plt.figure(figsize=(16, 12))
sns.heatmap(df_analysis[numeric_features].corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features", fontsize=16)
plt.tight_layout()
plt.savefig("../figures/correlation_matrix_numeric_features.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 7. Identify repayment variables with the clearest separation ******

df_analysis["target_numeric"] = df_analysis[TARGET_COL].astype(int)

pay_separation = []

for col in PAY_COLS:
    rates = df_analysis.groupby("target_numeric")[col].apply(lambda x: (x > 0).mean() * 100)

    non_default_rate = rates.get(0, np.nan)
    default_rate = rates.get(1, np.nan)

    pay_separation.append({"feature": col, "Non-default": non_default_rate, "Default": default_rate, "difference": abs(default_rate - non_default_rate),})

pay_separation = pd.DataFrame(pay_separation).sort_values("difference", ascending=False)

print("\n=== PAY Separation ===")
print(pay_separation.to_string(index=False))

payment_separation = (df_analysis.groupby("target_numeric")[PAYMENT_COLS].median().T.rename(columns={0: "Non-default", 1: "Default"}))
payment_separation["difference"] = abs(payment_separation["Default"] - payment_separation["Non-default"])
payment_separation = payment_separation.sort_values("difference", ascending=False)

print("\n=== Payment Amount Separation ===")
print(payment_separation.to_string())

bill_separation = (df_analysis.groupby("target_numeric")[BILL_COLS].median().T.rename(columns={0: "Non-default", 1: "Default"}))
bill_separation["difference"] = abs(bill_separation["Default"] - bill_separation["Non-default"])
bill_separation = bill_separation.sort_values("difference", ascending=False)

print("\n=== Bill Amount Separation ===")
print(bill_separation.to_string())

plot_configs = [{"data": pay_separation.melt(id_vars=["feature", "difference"], value_vars=["Non-default", "Default"], var_name="Default Status", value_name="Late Payment Rate",), "x": "feature", "y": "Late Payment Rate", "title": "Late Payment Rate by Repayment Status", "xlabel": "Repayment Status Variable", "ylabel": "Late Payment Rate (%)",}, {"data": payment_separation.reset_index().rename(columns={"index": "feature"}).melt(id_vars=["feature", "difference"], value_vars=["Non-default", "Default"], var_name="Default Status", value_name="Median Payment",), "x": "feature", "y": "Median Payment", "title": "Median Payment Amount by Default Status", "xlabel": "Payment Period", "ylabel": "Median Payment Amount",}, {"data": bill_separation.reset_index().rename(columns={"index": "feature"}).melt(id_vars=["feature", "difference"], value_vars=["Non-default", "Default"], var_name="Default Status", value_name="Median Bill",), "x": "feature", "y": "Median Bill", "title": "Median Bill Amount by Default Status", "xlabel": "Bill Period", "ylabel": "Median Bill Amount",},]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, config in zip(axes, plot_configs):
    sns.barplot(data=config["data"], x=config["x"], y=config["y"], hue="Default Status", ax=ax,)
    ax.set_title(config["title"], fontsize=14, fontweight="bold")
    ax.set_xlabel(config["xlabel"], fontsize=12)
    ax.set_ylabel(config["ylabel"], fontsize=12)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("../figures/repayment_bill_payment_separation.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

payment_trend = df_analysis.groupby("target_numeric")[PAYMENT_COLS].median().T
payment_trend.columns = ["Non-default", "Default"]

plt.figure(figsize=(10, 6))
sns.lineplot(data=payment_trend, markers=True)
plt.title("Trend of Median Payment Amount Over 6 Months by Default Status", fontsize=14)
plt.xlabel("Payment Period (1 = Most Recent, 6 = Oldest)")
plt.ylabel("Median Payment Amount")
plt.xticks(ticks=range(6), labels=["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],)
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/median_payment_trend_by_default.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 8. Engineered repayment and debt indicators ******

# Calculate repayment weights.
diff_dict = dict(zip(pay_separation["feature"], pay_separation["difference"]))
diffs = [diff_dict[col] for col in PAY_COLS]
total_diff = sum(diffs)

if total_diff == 0:
    weights = [1 / len(PAY_COLS)] * len(PAY_COLS)
else:
    weights = [difference / total_diff for difference in diffs]

# Give more weight to the most recent month.
emphasis_factor = 1.4
weights[0] *= emphasis_factor
weights = [weight / sum(weights) for weight in weights]

print("\n=== Automatically Calculated Repayment Weights ===")
for col, weight in zip(PAY_COLS, weights):
    print(f"{col}: {weight:.4f}")

weighted_ratio = pd.Series(0.0, index=df_analysis.index)

for i in range(6):
    ratio = df_analysis[PAYMENT_COLS[i]] / df_analysis[BILL_COLS[i]].replace(0, 1)
    ratio = ratio.clip(lower=0, upper=1)
    weighted_ratio += weights[i] * ratio

df_analysis["repayment_ratio"] = weighted_ratio

df_analysis["total_bill"] = df_analysis[BILL_COLS].sum(axis=1)
df_analysis["total_pay"] = df_analysis[PAYMENT_COLS].sum(axis=1)
df_analysis["debt_balance"] = df_analysis["total_bill"] - df_analysis["total_pay"]
df_analysis["debt_ratio"] = df_analysis["debt_balance"] / df_analysis["LIMIT_BAL"]

print("\n=== Weighted Repayment Ratio Distribution ===")
print(df_analysis["repayment_ratio"].describe())

group_repayment = (df_analysis.groupby(customer_group_columns, as_index=False, observed=True).agg(customer_count=("ID", "count"), mean_repayment_ratio=("repayment_ratio", "mean"), median_repayment_ratio=("repayment_ratio", "median"), default_rate=("target_numeric", "mean"), late_payment_rate=("had_late_payment", "mean"), median_total_bill=("total_bill", "median"), median_total_pay=("total_pay", "median"), mean_debt_balance=("debt_balance", "mean"), median_debt_balance=("debt_balance", "median"), mean_debt_ratio=("debt_ratio", "mean"), median_debt_ratio=("debt_ratio", "median"),).sort_values("median_repayment_ratio", ascending=False))

for col in customer_group_columns:
    group_detail_repayment = (df_analysis.groupby(col, as_index=False, observed=True).agg(customer_count=("ID", "count"), mean_repayment_ratio=("repayment_ratio", "mean"), median_repayment_ratio=("repayment_ratio", "median"), default_rate=("target_numeric", "mean"), late_payment_rate=("had_late_payment", "mean"), median_total_bill=("total_bill", "median"), median_total_pay=("total_pay", "median"), mean_debt_balance=("debt_balance", "mean"), median_debt_balance=("debt_balance", "median"), mean_debt_ratio=("debt_ratio", "mean"), median_debt_ratio=("debt_ratio", "median"),).query("customer_count >= 50").sort_values("median_repayment_ratio", ascending=False))

    print(f"\n=== Repayment Summary by {col} (Customer Count >= 50) ===")
    print(group_detail_repayment.to_string(index=False))

print("\n=== Combined Repayment Summary ===")
print(group_repayment.to_string(index=False))

print("\n=== Combined Repayment Summary (Customer Count >= 50) ===")
group_repayment_filtered = group_repayment.query("customer_count >= 50")
print(group_repayment_filtered.to_string(index=False))


#****** 9. Sensitivity analysis for the engineered repayment ratio ******

repayment_thresholds = [0.3, 0.5, 0.7, 0.9]
repayment_threshold_results = []

for threshold in repayment_thresholds:
    temp_df = df_analysis.copy()
    temp_df["risk_group_test"] = np.where(temp_df["repayment_ratio"] < threshold, "High Risk", "Low Risk",)

    high_risk_mask = temp_df["risk_group_test"] == "High Risk"
    high_risk_default_rate = temp_df.loc[high_risk_mask, "target_numeric"].mean()
    high_risk_count = high_risk_mask.sum()
    total_defaults = temp_df["target_numeric"].sum()
    high_risk_defaults = temp_df.loc[high_risk_mask, "target_numeric"].sum()

    repayment_threshold_results.append({"threshold": threshold, "high_risk_count": high_risk_count, "high_risk_default_rate": high_risk_default_rate, "pct_of_defaults_captured": (high_risk_defaults / total_defaults) * 100, "pct_of_customers_flagged": (high_risk_count / len(temp_df)) * 100,})

repayment_threshold_results_df = pd.DataFrame(repayment_threshold_results)

print("\n=== Repayment Ratio Sensitivity Analysis ===")
print(repayment_threshold_results_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.plot(repayment_threshold_results_df["threshold"], repayment_threshold_results_df["pct_of_defaults_captured"], marker="o", label="% of Defaults Captured",)
plt.plot(repayment_threshold_results_df["threshold"], repayment_threshold_results_df["pct_of_customers_flagged"], marker="s", label="% of Customers Flagged as High Risk",)
plt.xlabel("Repayment Ratio Threshold")
plt.ylabel("Percentage (%)")
plt.title("Sensitivity Analysis: Threshold vs. Detection and Flagging Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/repayment_ratio_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 10. Interaction analysis: PAY_0 and marital status ******

interaction_table = pd.crosstab([df_analysis["MARRIAGE"], df_analysis["PAY_0"]], df_analysis["default_label"], normalize="columns",) * 100

print("\n=== Interaction: PAY_0 and MARRIAGE vs Default ===")
print(interaction_table.round(2))

interaction_plot = interaction_table.reset_index()
interaction_plot.columns = ["MARRIAGE", "PAY_0", "Non-default", "Default"]

plt.figure(figsize=(10, 6))
sns.barplot(data=interaction_plot, x="PAY_0", y="Default", hue="MARRIAGE")
plt.title("Default Distribution by PAY_0 and MARRIAGE", fontsize=14)
plt.xlabel("PAY_0 Status (Most Recent Repayment Status)")
plt.ylabel("Default Group Share (%)")
plt.legend(title="MARRIAGE")
plt.tight_layout()
plt.savefig("../figures/pay0_marriage_default_interaction.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 11. PCA on payment amounts ******

payment_scaler = StandardScaler()
X_payment_scaled = payment_scaler.fit_transform(df_analysis[PAYMENT_COLS])

payment_pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_payment_pca = payment_pca.fit_transform(X_payment_scaled)

df_analysis["PC1"] = X_payment_pca[:, 0]
df_analysis["PC2"] = X_payment_pca[:, 1]

payment_loadings = pd.DataFrame(payment_pca.components_.T, columns=["PC1", "PC2"], index=PAYMENT_COLS,)

print("\n=== PCA Loadings for Payment Amounts ===")
print(payment_loadings.round(3))

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_payment_pca[:, 0], X_payment_pca[:, 1], c=df_analysis["target_numeric"], cmap="coolwarm", alpha=0.6, s=30,)
plt.colorbar(scatter, label="Default (1 = Yes, 0 = No)")

for i, (x_value, y_value) in enumerate(payment_pca.components_.T):
    plt.arrow(0, 0, x_value * 3, y_value * 3, head_width=0.1, head_length=0.1, fc="black", ec="black",)
    plt.text(x_value * 3.2, y_value * 3.2, PAYMENT_COLS[i], fontsize=10, ha="center", va="center", fontweight="bold",)

plt.xlabel(f"Principal Component 1 ({payment_pca.explained_variance_ratio_[0] * 100:.2f}% Variance)")
plt.ylabel(f"Principal Component 2 ({payment_pca.explained_variance_ratio_[1] * 100:.2f}% Variance)")
plt.title("Biplot: PCA of Payment Amounts with Feature Loadings")
plt.grid(True)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig("../figures/pca_payment_amounts_biplot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#****** 12. Prepare data for modeling and clustering ******

# Use original project features for modeling.
model_data = df_analysis.drop(columns=[id_col, "age_group", "default_label", "target_numeric", "late_payment_count", "had_late_payment", "repayment_ratio", "total_bill", "total_pay", "debt_balance", "debt_ratio", "PC1", "PC2",], errors="ignore",).copy()

nominal_cols = ["SEX", "EDUCATION", "MARRIAGE"]

continuous_cols = ["LIMIT_BAL", "AGE", *BILL_COLS, *PAYMENT_COLS]
ordinal_cols = PAY_COLS
numeric_cols = [*continuous_cols, *ordinal_cols]
smote_categorical_cols = [*ordinal_cols, *nominal_cols]

model_data[continuous_cols] = model_data[continuous_cols].astype(float)

X = model_data.drop(columns=[TARGET_COL])
y = model_data[TARGET_COL].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,)

print("\n=== Train/Test Split ===")
print(f"Train size: {len(X_train_raw)}")
print(f"Test size: {len(X_test_raw)}")


#****** 13. Preprocessing ******

preprocessor = ColumnTransformer(transformers=[("num", Pipeline(steps=[("winsorizer", Winsorizer(capping_method="quantiles", tail="both", fold=0.05),)
    , ("power", PowerTransformer(method="yeo-johnson")), ("scaler", RobustScaler()),]), continuous_cols,),
    ("ord", "passthrough", ordinal_cols,),
    ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),)]), nominal_cols,),])

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
X_test_df = pd.DataFrame(X_test, columns=feature_names, index=X_test_raw.index)

#****** Check distributions and outliers before preprocessing ******

distribution_plot(4, 5, X_train_raw, numeric_cols, "numeric_before_preprocessing_distribution.png")
boxplot_grid(4, 5, X_train_raw, numeric_cols, "numeric_before_preprocessing_boxplot.png")


#****** Check distributions and outliers after preprocessing ******
numeric_transformed_cols = [col for col in X_train_df.columns if col.startswith("num__") or col.startswith("ord__")]
# Before preprocessing
print("\n=== Test Normality Before Preprocessing with Shapiro-Wilk ===")
test_normality(X_train_raw, continuous_cols)
# After preprocessing
continuous_transformed_cols = [f"num__{col}" for col in continuous_cols]
print("\n=== Test Normality After Preprocessing with Shapiro-Wilk ===")
test_normality(X_train_df, continuous_transformed_cols)

distribution_plot(4, 5, X_train_df, numeric_transformed_cols, "numeric_after_preprocessing_distribution.png")
boxplot_grid(4, 5, X_train_df, numeric_transformed_cols, "numeric_after_preprocessing_boxplot.png")


print("\n=== Preprocessed Data Shapes ===")
print(f"Train shape: {X_train_df.shape}")
print(f"Test shape: {X_test_df.shape}")

# Transform full data with the fitted preprocessor.
X_full = preprocessor.transform(X)
X_full_df = pd.DataFrame(X_full, columns=feature_names, index=X.index)


#****** 14. Simple Logistic Regression baseline ******

baseline_log = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,)
baseline_log.fit(X_train, y_train)

baseline_test_pred = baseline_log.predict(X_test)
baseline_test_probs = baseline_log.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - SIMPLE BASELINE")
print("=" * 60)
print(f"Test Accuracy: {accuracy_score(y_test, baseline_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, baseline_test_probs):.4f}")
print(f"Average Precision: {average_precision_score(y_test, baseline_test_probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, baseline_test_pred, zero_division=0))

#****** Simple Baseline Logistic Regression threshold comparison ******

# Fixed thresholds are reported on the test set only for comparison.
baseline_thresholds = [0.3, 0.5, 0.7]
baseline_threshold_results = []

for threshold in baseline_thresholds:
    baseline_pred_threshold = apply_threshold(baseline_test_probs, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, baseline_pred_threshold).ravel()

    if threshold == 0.3:
        interpretation = "Max Detection (Aggressive)"
    elif threshold == 0.5:
        interpretation = "Standard (Default)"
    else:
        interpretation = "Reduce Operations (Conservative)"

    baseline_threshold_results.append({"Threshold": threshold, "Precision": precision_score(y_test, baseline_pred_threshold, zero_division=0), "Recall": recall_score(y_test, baseline_pred_threshold, zero_division=0), "F1-Score": f1_score(y_test, baseline_pred_threshold, zero_division=0), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Interpretation": interpretation,})

baseline_threshold_results_df = pd.DataFrame(baseline_threshold_results)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 4)

print("\n=== Simple Baseline Logistic Threshold Comparison ===")
print(baseline_threshold_results_df.round(4).to_string(index=False))


#****** 15. Optimized Logistic Regression with SMOTENC ******

# Keep part of the training data only for validation.
X_log_train_raw, X_log_val_raw, y_log_train, y_log_val = train_test_split(X_train_raw, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train,)

preprocessor_smote = ColumnTransformer(transformers=[("num", Pipeline(steps=[("winsorizer", Winsorizer(capping_method="quantiles", tail="both", fold=0.05),),
    ("power", PowerTransformer(method="yeo-johnson")), ("scaler", RobustScaler()),]), continuous_cols,),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), smote_categorical_cols,),],
    verbose_feature_names_out=False,)

smote_categorical_indices = list(range(len(continuous_cols), len(continuous_cols) + len(smote_categorical_cols)))

postprocessor_smote = ColumnTransformer(transformers=[("num", "passthrough", list(range(len(continuous_cols)))),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), smote_categorical_indices,),],
    verbose_feature_names_out=False,)


def logistic_objective(trial):
    c_value = trial.suggest_float("C", 0.001, 10, log=True)
    logistic_model = ImbPipeline(steps=[("preprocessor_smote", clone(preprocessor_smote)),
        ("smotenc", SMOTENC(categorical_features=smote_categorical_indices, random_state=RANDOM_STATE)),
        ("postprocessor_smote", clone(postprocessor_smote)),
        ("model", LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)),])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(logistic_model, X_log_train_raw, y_log_train, cv=skf, scoring="roc_auc").mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),)
study.optimize(logistic_objective, n_trials=20)
best_params = study.best_params

print("\n=== Best Logistic Regression Parameters ===")
print(best_params)
print(f"Best Cross-Validated ROC-AUC: {study.best_value:.4f}")

# Learn probability-based risk boundaries without oversampling.
risk_model_val = Pipeline(steps=[("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),])
risk_model_val.fit(X_log_train_raw, y_log_train)

y_val_probs = risk_model_val.predict_proba(X_log_val_raw)[:, 1]

# Risk boundaries are learned from validation probabilities.
# Low Risk = lower 50%, High Risk = top 20% of validation risk.
low_risk_threshold = float(np.quantile(y_val_probs, 0.50))
high_risk_threshold = float(np.quantile(y_val_probs, 0.80))

logistic_risk_bins = [0.0, low_risk_threshold, high_risk_threshold, 1.0]
logistic_risk_labels = ["Low", "Medium", "High"]

print("\n=== Logistic Risk Thresholds from Validation ===")
print(f"Low / Medium threshold: {low_risk_threshold:.4f}")
print(f"Medium / High threshold: {high_risk_threshold:.4f}")

validation_risk_groups = pd.cut(y_val_probs, bins=logistic_risk_bins, labels=logistic_risk_labels, include_lowest=True,)

validation_risk_summary = pd.DataFrame({"predicted_probability": y_val_probs, "risk_group": validation_risk_groups, "observed_default": y_log_val.to_numpy(),})

validation_risk_summary = (validation_risk_summary.groupby("risk_group", observed=True).agg(customer_count=("risk_group", "size"), mean_predicted_probability=("predicted_probability", "mean"), observed_default_rate=("observed_default", "mean"),).reset_index())

print("\n=== Validation Risk Group Summary ===")
print(validation_risk_summary.to_string(index=False))

# Refit the final SMOTENC logistic model on the complete training set.
model_log = ImbPipeline(steps=[("preprocessor_smote", clone(preprocessor_smote)),
    ("smotenc", SMOTENC(categorical_features=smote_categorical_indices, random_state=RANDOM_STATE)),
    ("postprocessor_smote", clone(postprocessor_smote)),
    ("model", LogisticRegression(**study.best_params, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)),])
model_log.fit(X_train_raw, y_train)

y_train_pred = model_log.predict(X_train_raw)
y_test_pred = model_log.predict(X_test_raw)
y_train_probs = model_log.predict_proba(X_train_raw)[:, 1]
y_test_probs = model_log.predict_proba(X_test_raw)[:, 1]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - OPTIMIZED WITH SMOTENC")
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_probs):.4f}")
print(f"Test Average Precision: {average_precision_score(y_test, y_test_probs):.4f}")

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred, zero_division=0))

cm_log = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-default", "Default"], yticklabels=["Non-default", "Default"],)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../figures/logistic_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

skf_log = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_model_log = ImbPipeline(steps=[("preprocessor_smote", clone(preprocessor_smote)),
    ("smotenc", SMOTENC(categorical_features=smote_categorical_indices, random_state=RANDOM_STATE)),
    ("postprocessor_smote", clone(postprocessor_smote)),
    ("model", LogisticRegression(C=best_params["C"], solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)),])

cross_val_log = cross_val_score(cv_model_log, X_train_raw, y_train, cv=skf_log, scoring="roc_auc")

print("\nLogistic Regression 5-Fold Cross-Validation Scores:")
print(cross_val_log)
print(f"Cross-Validation Mean: {cross_val_log.mean():.4f}")

#****** 16. Logistic Regression threshold comparison ******

# Fixed thresholds are reported on the test set only for comparison.
logistic_thresholds = [0.3, 0.5, 0.7]
logistic_threshold_results = []

for threshold in logistic_thresholds:
    y_pred_threshold = apply_threshold(y_test_probs, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    if threshold == 0.3:
        interpretation = "Max Detection (Aggressive)"
    elif threshold == 0.5:
        interpretation = "Standard (Default)"
    else:
        interpretation = "Reduce Operations (Conservative)"

    logistic_threshold_results.append({"Threshold": threshold, "Precision": precision_score(y_test, y_pred_threshold, zero_division=0), "Recall": recall_score(y_test, y_pred_threshold, zero_division=0), "F1-Score": f1_score(y_test, y_pred_threshold, zero_division=0), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Interpretation": interpretation,})

logistic_threshold_results_df = pd.DataFrame(logistic_threshold_results)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 4)

print("\n=== Logistic Threshold Comparison ===")
print(logistic_threshold_results_df.round(4).to_string(index=False))


#****** 17. Logistic Regression ROC and Precision-Recall curves ******

fpr_log, tpr_log, _ = roc_curve(y_test, y_test_probs)
auc_log = roc_auc_score(y_test, y_test_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC = {auc_log:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/logistic_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

precision_log, recall_log, _ = precision_recall_curve(y_test, y_test_probs)
avg_precision_log = average_precision_score(y_test, y_test_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall_log, precision_log, label=f"Average Precision = {avg_precision_log:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/logistic_precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 18. Logistic Regression coefficients and SHAP ******

logistic_model_fitted = model_log.named_steps["model"]
logistic_preprocessor_fitted = model_log.named_steps["preprocessor_smote"]
logistic_postprocessor_fitted = model_log.named_steps["postprocessor_smote"]
X_train_log_preprocessed = logistic_preprocessor_fitted.transform(X_train_raw)
X_train_log = logistic_postprocessor_fitted.transform(X_train_log_preprocessed)

logistic_onehot_fitted = logistic_postprocessor_fitted.named_transformers_["cat"]
logistic_ordinal_fitted = logistic_preprocessor_fitted.named_transformers_["cat"]
logistic_categorical_feature_names = []

for column, encoded_values, original_values in zip(smote_categorical_cols, logistic_onehot_fitted.categories_, logistic_ordinal_fitted.categories_):
    for encoded_value in encoded_values:
        original_value = original_values[int(encoded_value)]
        logistic_categorical_feature_names.append(f"{column}_{original_value}")

feature_names_log = np.asarray([*continuous_cols, *logistic_categorical_feature_names])

coefficient_df = pd.DataFrame({"feature": feature_names_log, "coefficient": logistic_model_fitted.coef_[0],})
coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
coefficient_df = coefficient_df.sort_values("abs_coefficient", ascending=False)

print("\n=== Top Logistic Regression Coefficients ===")
print(coefficient_df.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))
top_coefficients = coefficient_df.head(15).sort_values("coefficient")
sns.barplot(data=top_coefficients, x="coefficient", y="feature", orient="h")
plt.title("Top Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../figures/logistic_top_coefficients.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

logistic_shap_size = min(5000, len(X_train_log))
logistic_shap_rng = np.random.default_rng(RANDOM_STATE)
logistic_shap_indices = logistic_shap_rng.choice(len(X_train_log), size=logistic_shap_size, replace=False)
X_train_log_shap = X_train_log[logistic_shap_indices]

explainer_log = shap.LinearExplainer(logistic_model_fitted, X_train_log_shap)
shap_values_log = explainer_log.shap_values(X_train_log_shap)

shap_importance_log = pd.DataFrame({"feature": feature_names_log, "importance": np.abs(shap_values_log).mean(axis=0),}).sort_values("importance", ascending=False)

print("\n=== SHAP Feature Importance - Logistic Regression ===")
print(shap_importance_log.head(15).to_string(index=False))

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_log, X_train_log_shap, feature_names=feature_names_log, max_display=15, show=False,)
plt.title("SHAP Summary Plot - Logistic Regression", fontsize=14)
plt.tight_layout()
plt.savefig("../figures/logistic_shap_summary.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_log, X_train_log_shap, feature_names=feature_names_log, plot_type="bar", max_display=15, show=False,)
plt.title("SHAP Feature Importance - Logistic Regression", fontsize=14)
plt.tight_layout()
plt.savefig("../figures/logistic_shap_bar.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

logistic_top5 = shap_importance_log.head(5)["feature"].values
logistic_shap_direction = coefficient_df.set_index("feature").loc[logistic_top5, ["coefficient"]].reset_index()
logistic_shap_direction["direction"] = np.where(logistic_shap_direction["coefficient"] > 0, "Higher values increase predicted default risk", "Higher values decrease predicted default risk")

print("\n=== SHAP Direction for Top 5 Logistic Features ===")
print(logistic_shap_direction.to_string(index=False))


#****** 19. RFE with Logistic Regression ******

log_reg_rfe = LogisticRegression(C=best_params["C"], solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,)

rfe_log = RFE(estimator=log_reg_rfe, n_features_to_select=1, step=1,)
rfe_log.fit(X_train_log, y_train)

rfe_ranking_log = pd.DataFrame({"feature": feature_names_log, "ranking": rfe_log.ranking_,}).sort_values("ranking").reset_index(drop=True)

rfe_ranking_log["rank"] = range(1, len(rfe_ranking_log) + 1)

print("\n=== RFE Feature Ranking - Logistic Regression ===")
print(rfe_ranking_log[["rank", "feature", "ranking"]].to_string(index=False))


#****** 20. PCA, UMAP, and t-SNE comparison on full dataset ******

# Use a separate preprocessor for full-dataset visualization.
preprocessor_vis = clone(preprocessor)
X_preprocessed = preprocessor_vis.fit_transform(X)

target_for_visualization = y.to_numpy()

# PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_preprocessed)

# UMAP
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_STATE, n_jobs=1)
X_umap = umap_reducer.fit_transform(X_preprocessed)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
X_tsne = tsne.fit_transform(X_preprocessed)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=target_for_visualization, cmap="coolwarm", s=5, alpha=0.6)
axes[0].set_title("PCA")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=target_for_visualization, cmap="coolwarm", s=5, alpha=0.6)
axes[1].set_title("UMAP")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")

axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_for_visualization, cmap="coolwarm", s=5, alpha=0.6)
axes[2].set_title("t-SNE")
axes[2].set_xlabel("t-SNE1")
axes[2].set_ylabel("t-SNE2")

plt.tight_layout()
plt.savefig("../figures/pca_umap_tsne_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 21. Optional UMAP + GMM hyperparameter tuning ******

# Optional tuning block.
# Parameters from the original experiment.
#
# def cluster_objective(trial):
#     n_neighbors = trial.suggest_int("n_neighbors", 10, 80)
#     min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
#     n_components = trial.suggest_int("n_components", 2, 8)
#     covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag"])
#
#     umap_model = umap.UMAP(
#         n_neighbors=n_neighbors,
#         min_dist=min_dist,
#         n_components=n_components,
#         random_state=RANDOM_STATE,
#     )
#     X_umap_trial = umap_model.fit_transform(X_train_df)
#
#     gmm_model = GaussianMixture(
#         n_components=3,
#         covariance_type=covariance_type,
#         random_state=RANDOM_STATE,
#     )
#     labels_trial = gmm_model.fit_predict(X_umap_trial)
#
#     if len(set(labels_trial)) < 2:
#         return -1.0
#
#     return silhouette_score(X_umap_trial, labels_trial)
#
# cluster_study = optuna.create_study(direction="maximize")
# cluster_study.optimize(cluster_objective, n_trials=20)
# print(f"Best Silhouette: {cluster_study.best_value:.4f}")
# print(cluster_study.best_params)


#****** 22. UMAP + GMM clustering ******

umap_final = umap.UMAP(n_neighbors=44, min_dist=0.0027571032141944507, n_components=3, random_state=RANDOM_STATE, n_jobs=1,)

X_train_umap = umap_final.fit_transform(X_train_df)
X_test_umap = umap_final.transform(X_test_df)

gmm_final = GaussianMixture(n_components=3, covariance_type="full", random_state=RANDOM_STATE,)

train_labels = gmm_final.fit_predict(X_train_umap)
test_labels = gmm_final.predict(X_test_umap)

train_silhouette = silhouette_score(X_train_umap, train_labels, sample_size=min(5000, len(X_train_umap)), random_state=RANDOM_STATE)
test_silhouette = silhouette_score(X_test_umap, test_labels, sample_size=min(5000, len(X_test_umap)), random_state=RANDOM_STATE)

print("\n=== UMAP + GMM Clustering ===")
print(f"Train Silhouette Score: {train_silhouette:.4f}")
print(f"Test Silhouette Score: {test_silhouette:.4f}")

print("\nTrain cluster distribution:")
print(pd.Series(train_labels).value_counts().sort_index())

print("\nTest cluster distribution:")
print(pd.Series(test_labels).value_counts().sort_index())

X_train_clustered = X_train_df.copy()
X_test_clustered = X_test_df.copy()

X_train_clustered["cluster_label"] = train_labels
X_test_clustered["cluster_label"] = test_labels

train_cluster_default = pd.DataFrame({"cluster_label": train_labels, "default": y_train.to_numpy(),})

default_rates_train = train_cluster_default.groupby("cluster_label")["default"].mean()
sorted_clusters_train = default_rates_train.sort_values().index.tolist()

risk_map_train = {sorted_clusters_train[0]: "Low Risk", sorted_clusters_train[1]: "Medium Risk", sorted_clusters_train[2]: "High Risk",}

X_train_clustered["risk_label"] = X_train_clustered["cluster_label"].map(risk_map_train)
X_test_clustered["risk_label"] = X_test_clustered["cluster_label"].map(risk_map_train)

print("\nDefault rate per cluster (Train):")
for cluster in sorted_clusters_train:
    print(f"Cluster {cluster} ({risk_map_train[cluster]}): {default_rates_train[cluster]:.2%}")

test_cluster_default = pd.DataFrame({"cluster_label": test_labels, "default": y_test.to_numpy(),})
default_rates_test = test_cluster_default.groupby("cluster_label")["default"].mean()

X_test_clustered["risk_label_test"] = X_test_clustered["cluster_label"].map(risk_map_train)

print("\nDefault rate per cluster (Test, reporting only):")
for cluster in sorted_clusters_train:
    print(f"Cluster {cluster} ({risk_map_train[cluster]}): {default_rates_test.get(cluster, np.nan):.2%}")


#****** 23. Combine cluster labels with original modeling columns ******

df_full = model_data.copy()

full_cluster_labels = pd.Series(index=df_full.index, dtype="int64")
full_cluster_labels.loc[X_train_raw.index] = train_labels
full_cluster_labels.loc[X_test_raw.index] = test_labels

df_full["cluster_label"] = full_cluster_labels.astype(int)
df_full["risk_label"] = df_full["cluster_label"].map(risk_map_train)

print("\n=== Cluster Label Validation ===")
print(f"Rows in df_full: {len(df_full)}")
print(f"cluster_label missing values: {df_full['cluster_label'].isna().sum()}")
print(f"risk_label missing values: {df_full['risk_label'].isna().sum()}")
print("Risk mapping learned from Train:")
print(risk_map_train)

analysis_cols = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1", TARGET_COL,]

cluster_summary = df_full.groupby("cluster_label")[analysis_cols].agg(["mean", "median"]).round(2)
cluster_summary.columns = ["_".join(col) for col in cluster_summary.columns]
cluster_summary["cluster_size"] = df_full.groupby("cluster_label").size()
cluster_summary["risk_label"] = df_full.groupby("cluster_label")["risk_label"].first()
cluster_summary = cluster_summary.sort_values(f"{TARGET_COL}_mean")

cluster_summary_columns = ["cluster_size", "risk_label",] + [col for col in cluster_summary.columns if col not in ["cluster_size", "risk_label"]]
cluster_summary = cluster_summary[cluster_summary_columns]

print("\n=== Cluster Summary ===")
print(cluster_summary.to_string())


#****** 24. Cluster visualizations ******

umap_vis = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=RANDOM_STATE, n_jobs=1,)
X_vis = umap_vis.fit_transform(X_full_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter_clusters = axes[0].scatter(X_vis[:, 0], X_vis[:, 1], c=df_full["cluster_label"], cmap="Spectral", s=8, alpha=0.6,)
axes[0].set_title("Clusters in UMAP Space")
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
plt.colorbar(scatter_clusters, ax=axes[0], label="Cluster")

risk_num = df_full["risk_label"].map({"Low Risk": 0, "Medium Risk": 1, "High Risk": 2})
scatter_risk = axes[1].scatter(X_vis[:, 0], X_vis[:, 1], c=risk_num, cmap="coolwarm", s=8, alpha=0.6,)
axes[1].set_title("Risk Labels in UMAP Space")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
plt.colorbar(scatter_risk, ax=axes[1], label="Risk Level (0 = Low, 2 = High)")

plt.tight_layout()
plt.savefig("../figures/umap_clusters_and_risk_labels.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter_pay0 = axes[0].scatter(df_full["PAY_0"], df_full[TARGET_COL], c=df_full["cluster_label"], cmap="Spectral", s=20, alpha=0.6,)
axes[0].set_title("PAY_0 vs Default by Cluster")
axes[0].set_xlabel("PAY_0")
axes[0].set_ylabel("Default (1 = Yes)")
axes[0].grid(alpha=0.3)
plt.colorbar(scatter_pay0, ax=axes[0], label="Cluster")

scatter_pay_amt1 = axes[1].scatter(df_full["PAY_AMT1"], df_full[TARGET_COL], c=df_full["cluster_label"], cmap="Spectral", s=20, alpha=0.6,)
axes[1].set_title("PAY_AMT1 vs Default by Cluster")
axes[1].set_xlabel("PAY_AMT1")
axes[1].set_ylabel("Default (1 = Yes)")
axes[1].grid(alpha=0.3)
plt.colorbar(scatter_pay_amt1, ax=axes[1], label="Cluster")

plt.tight_layout()
plt.savefig("../figures/cluster_scatter_pay0_payment_default.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 25. Save clustering outputs ******

train_export = df_full.loc[X_train_raw.index].copy()
test_export = df_full.loc[X_test_raw.index].copy()

with pd.ExcelWriter("../cluster_dataset/clusters_train_test_final.xlsx", mode="w") as writer:
    train_export.to_excel(writer, sheet_name="Train", index=False)
    test_export.to_excel(writer, sheet_name="Test", index=False)

cluster_summary.to_excel("../cluster_dataset/cluster_summary.xlsx")

print("\n=== Saved Clustering Files ===")
print(f"Train rows: {len(train_export)}")
print(f"Test rows: {len(test_export)}")
print("clusters_train_test_final.xlsx")
print("cluster_summary.xlsx")


#****** 26. Random Forest model to predict GMM cluster labels ******


df_rf = df_full.copy()
df_rf = df_rf.drop(columns=[TARGET_COL, "risk_label"], errors="ignore")

y_rf = df_rf["cluster_label"].astype(int)
X_rf = df_rf.drop(columns=["cluster_label"])

# x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_rf,)

x_train_rf = X_rf.loc[X_train_raw.index].copy()
x_test_rf = X_rf.loc[X_test_raw.index].copy()

y_train_rf = y_rf.loc[X_train_raw.index].copy()
y_test_rf = y_rf.loc[X_test_raw.index].copy()

print("\n=== Random Forest Train/Test Split ===")
print(f"Train shape: {x_train_rf.shape}, Train target shape: {y_train_rf.shape}")
print(f"Test shape: {x_test_rf.shape}, Test target shape: {y_test_rf.shape}")

preprocessor_rf = ColumnTransformer(transformers=[("num", "passthrough", continuous_cols),
            ("ord", "passthrough", ordinal_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols,),])

#****** Random Forest Optuna ******

def rf_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE, n_jobs=-1)

    rf_pipeline_trial = Pipeline(steps=[("preprocessor", clone(preprocessor_rf)), ("model", rf_model),])

    skf_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    return cross_val_score(rf_pipeline_trial, x_train_rf, y_train_rf, cv=skf_rf, scoring="accuracy").mean()


study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_rf.optimize(rf_objective, n_trials=20)

best_params_rf = study_rf.best_params

print("\n=== Best Random Forest Parameters ===")
print(best_params_rf)
print(f"Best Cross-Validated Accuracy: {study_rf.best_value:.4f}")


model_rf = Pipeline(steps=[("preprocessor", preprocessor_rf),
        ("model",RandomForestClassifier(
        **study_rf.best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),])

model_rf.fit(x_train_rf, y_train_rf)
print("\nRandom Forest model trained successfully.")


#****** 27. Random Forest evaluation on cluster labels ******

y_pred_rf = model_rf.predict(x_test_rf)
acc_rf = accuracy_score(y_test_rf, y_pred_rf)

rf_classes = model_rf.named_steps["model"].classes_
rf_class_names = [risk_map_train.get(int(cluster), f"Cluster {cluster}") for cluster in rf_classes]

print(f"\nRandom Forest Test Accuracy: {acc_rf:.4f}")
print("\nClassification Report (Test):")
print(classification_report(y_test_rf, y_pred_rf, labels=rf_classes, target_names=rf_class_names, zero_division=0,))

print(f"Train Accuracy: {model_rf.score(x_train_rf, y_train_rf):.4f}")
print(f"Test Accuracy: {model_rf.score(x_test_rf, y_test_rf):.4f}")

cm_rf = confusion_matrix(y_test_rf, y_pred_rf, labels=rf_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=rf_class_names, yticklabels=rf_class_names,)
plt.title(f"Confusion Matrix - Random Forest Clusters (Acc: {acc_rf:.4f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../figures/random_forest_cluster_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

skf_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cross_val_rf = cross_val_score(model_rf, x_train_rf, y_train_rf, cv=skf_rf, scoring="accuracy",)

print("\nRandom Forest 5-Fold Cross-Validation Scores:")
print(cross_val_rf)
print(f"Cross-Validation Mean: {cross_val_rf.mean():.4f}")


#****** 28. Random Forest threshold analysis for the High Risk cluster ******

high_risk_cluster = next(cluster for cluster, risk_label in risk_map_train.items() if risk_label == "High Risk")

high_risk_class_index = list(rf_classes).index(high_risk_cluster)

y_test_probs_rf = model_rf.predict_proba(x_test_rf)[:, high_risk_class_index]
y_true_binary_rf = (y_test_rf == high_risk_cluster).astype(int)

rf_thresholds = [0.3, 0.5, 0.7]
rf_threshold_results = []

for threshold in rf_thresholds:
    y_pred_threshold = apply_threshold(y_test_probs_rf, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true_binary_rf, y_pred_threshold).ravel()

    if threshold == 0.3:
        interpretation = "Max Detection (Aggressive)"
    elif threshold == 0.5:
        interpretation = "Standard (Default)"
    else:
        interpretation = "Reduce Operations (Conservative)"

    rf_threshold_results.append({"Threshold": threshold, "Precision": precision_score(y_true_binary_rf, y_pred_threshold, zero_division=0), "Recall": recall_score(y_true_binary_rf, y_pred_threshold, zero_division=0), "F1-Score": f1_score(y_true_binary_rf, y_pred_threshold, zero_division=0), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Interpretation": interpretation,})

rf_threshold_results_df = pd.DataFrame(rf_threshold_results)

print("\n=== Random Forest Threshold Comparison - High Risk Cluster ===")
print(rf_threshold_results_df.round(4).to_string(index=False))


#****** 29. Random Forest ROC and Precision-Recall curves ******

fpr_rf, tpr_rf, _ = roc_curve(y_true_binary_rf, y_test_probs_rf)
auc_rf = roc_auc_score(y_true_binary_rf, y_test_probs_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest High Risk Cluster")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/random_forest_high_risk_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

precision_rf, recall_rf, _ = precision_recall_curve(y_true_binary_rf, y_test_probs_rf)
avg_precision_rf = average_precision_score(y_true_binary_rf, y_test_probs_rf)

plt.figure(figsize=(8, 6))
plt.plot(recall_rf, precision_rf, label=f"Average Precision = {avg_precision_rf:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest High Risk Cluster")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/random_forest_high_risk_precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#****** 30. SHAP analysis for Random Forest ******

X_train_rf_transformed = model_rf.named_steps["preprocessor"].transform(x_train_rf)
X_test_rf_transformed = model_rf.named_steps["preprocessor"].transform(x_test_rf)
feature_names_rf = model_rf.named_steps["preprocessor"].get_feature_names_out()

print("\n=== Random Forest Transformed Data ===")
print(f"X_train_rf shape: {X_train_rf_transformed.shape}")
print(f"Number of features: {len(feature_names_rf)}")


rf_model_shap = RandomForestClassifier(n_estimators=best_params_rf["n_estimators"],
                                       max_depth=best_params_rf["max_depth"],
                                       min_samples_split=best_params_rf["min_samples_split"],
                                       min_samples_leaf=best_params_rf["min_samples_leaf"],
                                       random_state=RANDOM_STATE, n_jobs=-1)
rf_model_shap.fit(X_train_rf_transformed, y_train_rf)

rf_shap_size = min(2000, len(X_train_rf_transformed))
rf_shap_rng = np.random.default_rng(RANDOM_STATE)
rf_shap_indices = rf_shap_rng.choice(len(X_train_rf_transformed), size=rf_shap_size, replace=False)
X_train_rf_shap = X_train_rf_transformed[rf_shap_indices]

explainer_rf = shap.TreeExplainer(rf_model_shap)
shap_values_rf = explainer_rf.shap_values(X_train_rf_shap)

high_risk_shap_index = list(rf_model_shap.classes_).index(high_risk_cluster)

if isinstance(shap_values_rf, list):
    shap_values_high = shap_values_rf[high_risk_shap_index]
else:
    shap_values_high = shap_values_rf[:, :, high_risk_shap_index]

print("SHAP original shape:", np.shape(shap_values_rf))
print("SHAP High Risk shape:", shap_values_high.shape)

shap_importance_rf = pd.DataFrame({"feature": feature_names_rf, "importance": np.abs(shap_values_high).mean(axis=0),}).sort_values("importance", ascending=False)

print("\n=== SHAP Feature Importance - Random Forest High Risk Class ===")
print(shap_importance_rf.head(25).to_string(index=False))

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_high, X_train_rf_shap, feature_names=feature_names_rf, max_display=15, show=False,)
plt.title("SHAP Summary Plot - Random Forest High Risk Class", fontsize=14)
plt.tight_layout()
plt.savefig("../figures/random_forest_shap_summary_high_risk.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_high, X_train_rf_shap, feature_names=feature_names_rf, plot_type="bar", max_display=15, show=False,)
plt.title("SHAP Feature Importance - Random Forest High Risk Class", fontsize=14)
plt.tight_layout()
plt.savefig("../figures/random_forest_shap_bar_high_risk.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

rf_top5 = shap_importance_rf.head(5)["feature"].values
rf_shap_direction_rows = []

for feature in rf_top5:
    feature_index = list(feature_names_rf).index(feature)
    feature_values = X_train_rf_shap[:, feature_index]
    feature_shap_values = shap_values_high[:, feature_index]

    if np.std(feature_values) == 0 or np.std(feature_shap_values) == 0:
        correlation = np.nan
        direction = "No clear direction"
    else:
        correlation = np.corrcoef(feature_values, feature_shap_values)[0, 1]
        direction = "Higher values increase High Risk cluster probability" if correlation > 0 else "Higher values decrease High Risk cluster probability"

    rf_shap_direction_rows.append({"feature": feature, "value_shap_correlation": correlation, "direction": direction,})

rf_shap_direction = pd.DataFrame(rf_shap_direction_rows)

print("\n=== SHAP Direction for Top 5 Random Forest Features ===")
print(rf_shap_direction.to_string(index=False))


#****** 31. RFE with Random Forest ******

rf_model_rfe = RandomForestClassifier(n_estimators=best_params_rf["n_estimators"],
                                      max_depth=best_params_rf["max_depth"],
                                      min_samples_split=best_params_rf["min_samples_split"],
                                      min_samples_leaf=best_params_rf["min_samples_leaf"],
                                      random_state=RANDOM_STATE, n_jobs=-1)
rfe_rf = RFE(estimator=rf_model_rfe, n_features_to_select=1, step=1,)
rfe_rf.fit(X_train_rf_transformed, y_train_rf)

rfe_ranking_rf = pd.DataFrame({"feature": feature_names_rf, "ranking": rfe_rf.ranking_,}).sort_values("ranking").reset_index(drop=True)

rfe_ranking_rf["rank"] = range(1, len(rfe_ranking_rf) + 1)

print("\n=== RFE Feature Ranking - Random Forest ===")
print(rfe_ranking_rf[["rank", "feature", "ranking"]].to_string(index=False))


#****** 32. Logistic risk groups - PDF method ******

# Use the non-oversampled logistic model for default probabilities.
probs_log = baseline_log.predict_proba(X_full)[:, 1]

risk_groups_log = pd.cut(probs_log, bins=logistic_risk_bins, labels=logistic_risk_labels, include_lowest=True,)

# Main risk output required by the project.
df_log = pd.DataFrame({"predicted_probability": probs_log, "risk_group": risk_groups_log,})

df_log.to_csv("../outputs/risk_groups_logistic.csv", index=False)



# Report the risk groups on the untouched test set.
test_risk_groups_log = pd.cut(baseline_test_probs, bins=logistic_risk_bins, labels=logistic_risk_labels, include_lowest=True,)

logistic_test_risk = pd.DataFrame({"predicted_probability": baseline_test_probs, "risk_group": test_risk_groups_log, "observed_default": y_test.to_numpy(),})

logistic_risk_summary = (logistic_test_risk.groupby("risk_group", observed=True).agg(customer_count=("risk_group", "size"), mean_predicted_probability=("predicted_probability", "mean"), observed_default_rate=("observed_default", "mean"),).reset_index())

print("\n=== Logistic Risk Group Summary - Test ===")
print(logistic_risk_summary.to_string(index=False))

logistic_risk_summary.to_csv("../outputs/logistic_risk_summary_test.csv", index=False)

logistic_threshold_output = pd.DataFrame({"boundary": ["Low/Medium", "Medium/High"], "threshold": [low_risk_threshold, high_risk_threshold], "validation_rule": ["50th percentile of validation probability", "80th percentile of validation probability",],})
logistic_threshold_output.to_csv("../outputs/logistic_risk_thresholds.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=logistic_risk_summary, x="risk_group", y="observed_default_rate", order=logistic_risk_labels,)
plt.title("Observed Default Rate by Logistic Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Observed Default Rate")
plt.tight_layout()
plt.savefig("../figures/logistic_risk_group_default_rate.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("risk_groups_logistic.csv saved!")
print("logistic_risk_summary_test.csv saved!")
print("logistic_risk_thresholds.csv saved!")


#****** 33. Cluster risk groups - original method ******

# Keep the original cluster-based risk logic as an extra analysis.
cluster_risk_bins = [0, 0.2, 0.5, 1.0]
cluster_risk_labels = ["Low", "Medium", "High"]

probs_rf = model_rf.predict_proba(X_rf)[:, high_risk_class_index]
risk_groups_rf = pd.cut(probs_rf, bins=cluster_risk_bins, labels=cluster_risk_labels, include_lowest=True,)

df_rf_out = pd.DataFrame({"probability_high_risk_cluster_membership": probs_rf, "cluster_membership_risk_group": risk_groups_rf,})

df_rf_out.to_csv("../outputs/risk_groups_rf_cluster.csv", index=False)
print("risk_groups_rf_cluster.csv saved!")
plt.close('all')


f = open("../terminal_output/terminal_output.txt", "w", encoding="utf-8")
f.write("\n".join(all_outputs))
f.close()