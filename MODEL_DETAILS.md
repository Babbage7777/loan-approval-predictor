# Model Details

## 🧠 Overview
This project implements a **Loan Approval Prediction System** using an optimized **XGBoost classifier**. It provides loan eligibility predictions based on user input, which includes both categorical and numerical features. The system uses saved preprocessing artifacts to ensure consistent predictions.

---

## 📊 Dataset

- **Source:** `loan_data_set.csv` (based on the standard Loan Prediction dataset)
- **Target Variable:** `Loan_Status` (`Y` for approved, `N` for rejected)
- **Features Used:**
  - **Categorical:** Gender, Married, Dependents, Education, Self_Employed, Property_Area
  - **Numerical:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

---

## 🔍 Data Preprocessing

- **Missing Values:**
  - Categorical: Filled using mode
  - Numerical: Filled using median
- **Categorical Handling:**
  - Label encoding for binary fields (e.g., Gender, Married)
  - One-Hot Encoding for multiclass (e.g., Property_Area)
- **Numerical Scaling:**
  - Standard scaling applied using saved mean and standard deviation
- **Special Cases:**
  - `Dependents` feature mapped (e.g., "3+" → 3)
  - Unseen or missing categorical levels default to most common (mode)

---

## ⚙️ Model Configuration

- **Model:** `XGBClassifier` from `xgboost` library
- **Objective:** `binary:logistic`
- **Evaluation Metric:** `logloss`
- **Random State:** 42

### 🔧 Hyperparameter Tuning (via GridSearchCV)
The following grid was used:
- `max_depth`: 3, 4, 5  
- `learning_rate`: 0.01, 0.1, 0.2  
- `n_estimators`: 50, 100, 200  
- `gamma`: 0, 0.1, 0.2  
- `subsample`: 0.8, 0.9, 1.0  
- `colsample_bytree`: 0.8, 0.9, 1.0  

### 🏆 Best Parameters Example Output
```text
Best parameters found: {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100, 'gamma': 0, 'subsample': 0.9, 'colsample_bytree': 0.8}
Best ROC AUC score: 0.92
