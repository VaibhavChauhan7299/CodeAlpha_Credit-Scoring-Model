# ==========================================
# CREDIT SCORING MODEL
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ==========================================
# STEP 1: FIND PROJECT FOLDER
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

print("Project folder:")
print(BASE_DIR)


# ==========================================
# STEP 2: FIND DATASET
# ==========================================

DATA_PATH = BASE_DIR / "data" / "credit_data.csv"

print("\nDataset location:")
print(DATA_PATH)


# ==========================================
# STEP 3: LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# STEP 4: DISPLAY FIRST 5 ROWS
# ==========================================

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# STEP 5: DATASET SIZE
# ==========================================

print("\nDataset shape:")
print(df.shape)


# ==========================================
# STEP 6: COLUMN NAMES
# ==========================================

print("\nColumn names:")
print(df.columns.tolist())


# ==========================================
# STEP 7: DATA TYPES
# ==========================================

print("\nData types:")
print(df.dtypes)


# ==========================================
# STEP 8: MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# STEP 9: DUPLICATE ROWS
# ==========================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================
# STEP 10: DATA CLEANING
# ==========================================

print("\nNumerical columns summary:")
print(df.describe())


print("\nCategorical columns:")

categorical_columns = df.select_dtypes(include="object").columns

print(categorical_columns.tolist())


print("\nUnique values in categorical columns:")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())


print("\nTarget values:")
print(df["credit_risk"].unique())


# ==========================================
# STEP 11: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# 4.1 Credit Risk Distribution

plt.figure(figsize=(6, 4))

sns.countplot(x="credit_risk", data=df)

plt.title("Credit Risk Distribution")
plt.xlabel("Credit Risk")
plt.ylabel("Number of Customers")

plt.show()

# 4.2 Credit Amount vs Credit Risk

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="credit_risk",
    y="credit_amount",
    data=df
)

plt.title("Credit Amount vs Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Credit Amount")

plt.show()

# 4.3 Age vs Credit Risk

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="credit_risk",
    y="age",
    data=df
)

plt.title("Age vs Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Age")

plt.show()

# 4.4 Duration vs Credit Risk

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="credit_risk",
    y="duration",
    data=df
)

plt.title("Loan Duration vs Credit Risk")
plt.xlabel("Credit Risk")
plt.ylabel("Duration (months)")

plt.show()

# 4.5 Correlation Heatmap

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation = df[numerical_columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()