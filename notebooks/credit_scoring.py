# ==========================================
# CREDIT SCORING MODEL
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

# ==========================================
# STEP 5: FEATURE ENGINEERING
# ==========================================

X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# Identify numerical and categorical features

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# ==========================================
# STEP 6: TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# ==========================================
# STEP 7: FEATURE SCALING & PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# Fit ONLY on training data

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

# ==========================================
# STEP 8: LOGISTIC REGRESSION
# ==========================================

# Create the model

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model

logistic_model.fit(
    X_train_processed,
    y_train
)


print("\nLogistic Regression model trained successfully.")

# ==========================================
# STEP 9: DECISION TREE
# ==========================================

# Create the Decision Tree model

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)


# Train the model

decision_tree_model.fit(
    X_train_processed,
    y_train
)

print("\nDecision Tree model trained successfully.")

# ==========================================
# STEP 10: RANDOM FOREST
# ==========================================

# Create the Random Forest model

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the Random Forest model

random_forest_model.fit(
    X_train_processed,
    y_train
)


print("\nRandom Forest model trained successfully.")