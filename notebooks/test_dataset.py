import pandas as pd
from pathlib import Path


# ==========================================
# LOAD GERMAN CREDIT DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "german_credit" / "german.data"

print("Dataset path:")
print(data_path)

# Column names from the UCI German Credit dataset
columns = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk"
]


# Read dataset
df = pd.read_csv(
    data_path,
    sep=r"\s+",
    header=None,
    names=columns
)


# Convert target:
# 1 = Good credit
# 2 = Bad credit
#
# We change it to:
# 1 = Good
# 0 = Bad

df["credit_risk"] = df["credit_risk"].map({
    1: 1,
    2: 0
})


# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nTarget distribution:")
print(df["credit_risk"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# SAVE AS CSV
# ==========================================

output_path = BASE_DIR / "data" / "credit_data.csv"

df.to_csv(output_path, index=False)

print("\nDataset successfully saved!")
print(output_path)