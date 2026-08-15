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