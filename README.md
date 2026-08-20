# Credit Scoring Model

## 📌 Project Overview

This project is a Machine Learning-based Credit Scoring Model developed to predict an individual's creditworthiness using historical financial information.

The system analyzes customer and financial features and predicts the credit risk category using classification algorithms.

The following Machine Learning models were implemented and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Based on the evaluation results, Random Forest achieved the best overall performance.

---

## 🎯 Objective

The main objective of this project is to predict an individual's creditworthiness based on historical financial data.

The project uses classification algorithms to analyze customer information such as:

- Checking account status
- Credit duration
- Credit history
- Purpose of credit
- Credit amount
- Savings account
- Employment status
- Installment rate
- Age
- Housing
- Existing credits
- Job
- Other financial information

---

## 📊 Dataset

The project uses the German Credit dataset.

Dataset information:

- Total Records: 1000
- Total Columns: 21
- Input Features: 20
- Target Variable: `credit_risk`

The dataset contains both numerical and categorical features.

---

## ⚙️ Machine Learning Workflow

The project follows the following Machine Learning pipeline:

1. Import required libraries
2. Load the dataset
3. Inspect the dataset
4. Check missing values
5. Check duplicate records
6. Perform Exploratory Data Analysis
7. Identify numerical and categorical features
8. Separate features and target variable
9. Split the data into training and testing sets
10. Preprocess numerical and categorical data
11. Train Logistic Regression model
12. Train Decision Tree model
13. Train Random Forest model
14. Evaluate model performance
15. Compare all models
16. Select the best-performing model
17. Save the trained model
18. Load the saved model
19. Make predictions for new customer data

---

## 🤖 Models Used

### 1. Logistic Regression

Logistic Regression is a classification algorithm used to predict the probability of a binary outcome.

### 2. Decision Tree

A Decision Tree makes predictions by splitting data into different branches based on feature values.

### 3. Random Forest

Random Forest combines multiple Decision Trees and makes a final prediction based on the combined results of those trees.

Random Forest achieved the best performance in this project.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 70.50% | 77.55% | 81.43% | 79.44% |
| Decision Tree | 67.50% | 77.37% | 75.71% | 76.53% |
| **Random Forest** | **74.50%** | **78.34%** | **87.86%** | **82.83%** |

### 🏆 Best Model

**Random Forest** was selected as the best-performing model based on the evaluation results.

---

## 📏 Evaluation Metrics

The following metrics were used to evaluate the models:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 📁 Project Structure

```text
Credit-Scoring-Model/
│
├── data/
│   └── credit_data.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── credit_scoring.py
│
├── src/
│
├── README.md
│
└── requirements.txt