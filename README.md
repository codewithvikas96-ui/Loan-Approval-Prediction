# 🏦 Loan Approval Prediction
### Logistic Regression vs KNN vs Naive Bayes — A Classification Study

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Models](https://img.shields.io/badge/Models-3-purple)

---

## 📌 Project Overview

This project predicts whether a **loan application will be approved or rejected** based on applicant demographics, financial health, and loan details. Three classification models were trained, debugged, and iteratively improved:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Gaussian Naive Bayes**

The project went through **two full iterations**: a baseline run (which surfaced several real-world ML pitfalls) and a fixed/improved run with feature engineering, class imbalance handling, and proper preprocessing.

---

## 📂 Project Structure

```
Loan-Approval-Prediction/
│
├── loan_approval_data.csv                       # Raw dataset (1000 rows, 20 columns)
├── loan_approved_cleaned.csv                     # Cleaned + encoded dataset
├── creditWiseLoan_cleaning_preprocessing.ipynb    # Cleaning & encoding
├── EDA.ipynb                                      # Exploratory Data Analysis
├── ModelTraining.ipynb                            # Model training & evaluation
└── README.md
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `Applicant_ID` | int | Unique applicant identifier (dropped before modeling) |
| `Applicant_Income` | float | Primary applicant's income |
| `Coapplicant_Income` | float | Co-applicant's income |
| `Employment_Status` | object | Salaried / Self-employed |
| `Age` | float | Applicant's age |
| `Marital_Status` | object | Married / Single |
| `Dependents` | float | Number of dependents |
| `Credit_Score` | float | Applicant's credit score |
| `Existing_Loans` | float | Number of existing loans |
| `DTI_Ratio` | float | Debt-to-income ratio |
| `Savings` | float | Savings amount |
| `Collateral_Value` | float | Value of collateral offered |
| `Loan_Amount` | float | Requested loan amount |
| `Loan_Term` | float | Loan duration |
| `Loan_Purpose` | object | Home / Car / Education / Personal |
| `Property_Area` | object | Urban / Semiurban / Rural |
| `Education_Level` | object | Graduate / Non-graduate |
| `Gender` | object | Male / Female |
| `Employer_Category` | object | Government / MNC / Private / Unemployed |
| `Loan_Approved` | object | 🎯 **Target** — Yes / No |

**Shape:** 1000 rows × 20 columns | **Missing values present** (handled via imputation)

---

## 🔧 Data Cleaning & Preprocessing

1. **Missing value imputation**
   - Numeric columns → `SimpleImputer(strategy='mean')`
   - Categorical columns → `SimpleImputer(strategy='most_frequent')`
2. **Label Encoding** — `Marital_Status`, `Education_Level`, `Gender`, `Loan_Approved`
3. **One-Hot Encoding** — `Employment_Status`, `Loan_Purpose`, `Property_Area`, `Employer_Category` (`drop_first=True`)
4. **Dropped** `Applicant_ID` (non-predictive identifier)
5. Saved as `loan_approved_cleaned.csv`

---

## 📈 Exploratory Data Analysis (Highlights)

EDA was conducted across 6 thematic groups: target balance, demographics, financial health, loan details, employment/property, and correlation analysis.

**Key findings:**
- The target class is **imbalanced** — roughly 70% rejected vs 30% approved
- `Credit_Score` showed the strongest correlation with approval
- Applicants with **high DTI Ratio** and **low Credit Score** were disproportionately rejected
- `Applicant_Income` distribution is right-skewed with some outliers

---

## 🤖 Model Training — Iteration 1 (Baseline)

### Setup
- 80/20 train-test split, `random_state=42`
- `StandardScaler` applied to all numeric features
- All three models trained on the same fully-encoded dataset

### Results

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.75 | 0.77 | 0.86 |
| KNN (k=20, tuned) | 0.71 | 0.48 | 0.57 | 0.78 |
| **Gaussian Naive Bayes** | **0.00** | **0.00** | **0.00** | 0.70 |

### 🔴 Critical bug discovered

Naive Bayes predicted **every single test case as "Rejected"** — confusion matrix was `[[139, 0], [61, 0]]`. Investigation revealed multiple root causes:

1. **Data leakage** — in a later cell, the scaler was accidentally called as `scaler.fit_transform(X_test)` instead of `scaler.transform(X_test)`, fitting a *new* scaler on test data instead of reusing the training distribution.
2. **No class imbalance handling** — with ~70/30 class split and no `class_weight` or stratified split, weaker models defaulted to the majority class.
3. **GaussianNB violated its own assumptions** — one-hot encoded binary columns (0/1) do not follow a Gaussian distribution, which broke Naive Bayes' core likelihood calculation.

---

## 🛠️ Model Training — Iteration 2 (Fixed + Improved)

### Fixes applied

| Issue | Fix |
|---|---|
| Data leakage in scaling | `scaler.fit_transform(X_train)` → `scaler.transform(X_test)` (never fit on test) |
| Class imbalance | `train_test_split(..., stratify=y)` + `LogisticRegression(class_weight='balanced')` |
| GaussianNB Gaussian-assumption violation | Trained only on **continuous numeric features**, excluding one-hot/binary columns |
| KNN tuning metric | `GridSearchCV` scoring changed from `precision` → `f1` (balances both error types) |
| Decision threshold | Added probability-based prediction (`threshold = 0.4`) as an option alongside default 0.5 |

### New engineered features

```python
df['Total_Income']          = df['Applicant_Income'] + df['Coapplicant_Income']
df['Loan_to_Value_Ratio']   = df['Loan_Amount'] / df['Collateral_Value']
df['High_DTI_Flag']         = np.where(df['DTI_Ratio'] > 0.4, 1, 0)
df['EMI_Estimate']          = df['Loan_Amount'] / df['Loan_Term']
df['EMI_to_Income_Ratio']   = df['EMI_Estimate'] / df['Total_Income']
```

These mirror real underwriting logic — affordability (EMI-to-income) and risk (loan-to-value) ratios that banks use in practice.

### Final Results (after fixes + engineered features)

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression (`class_weight='balanced'`) | 0.65 | **0.93** | 0.77 | 0.83 |
| KNN (k=11, tuned on F1) | 0.63 | 0.65 | 0.64 | 0.78 |
| **Gaussian Naive Bayes** (continuous features only) | **0.74** | 0.72 | **0.73** | **0.84** |

### Naive Bayes recovery — before vs after

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Iteration 1 (broken — scaler leak + full feature set) | 0.00 | 0.00 | 0.00 |
| After fixing data leakage | 0.43 | 1.00 | 0.60 |
| After restricting to continuous features only | 0.70 | 0.72 | 0.71 |
| After adding engineered features | **0.74** | 0.72 | **0.73** |

A model that predicted nothing correctly was brought to a usable, competitive classifier purely through pipeline fixes — no architecture change.

---

## 💡 Key Learnings

1. **Always `fit` on train, `transform` on test** — fitting a scaler/encoder on test data is a silent but serious leakage bug, and it can fully break weaker models like Naive Bayes
2. **A 0.00 precision score is a debugging signal, not a tuning problem** — check the confusion matrix before reaching for hyperparameters
3. **Naive Bayes needs feature selection** — it assumes Gaussian-distributed inputs; one-hot encoded categorical columns violate this and can quietly destroy performance
4. **`stratify=y` should be a default habit** for any classification split, not an afterthought
5. **`class_weight='balanced'` is a one-line fix** for imbalanced binary classification and meaningfully shifts the precision/recall trade-off
6. **Encoding strategy affects Naive Bayes more than other models** — `LabelEncoder` on nominal (non-ordinal) categorical columns like `Marital_Status` or `Gender` can introduce a fake ordinal relationship; `OneHotEncoder`/`pd.get_dummies()` is the safer default for nominal binary/multi-class columns, even when there are only two categories
7. **Feature engineering should mirror domain logic** — ratios like `EMI_to_Income_Ratio` and `Loan_to_Value_Ratio` reflect how lenders actually assess risk, which is why they improved every model that used them

---

## 🚀 Tech Stack

| Tool | Usage |
|---|---|
| Python 3.10 | Core language |
| Pandas / NumPy | Data loading, cleaning, feature engineering |
| Scikit-Learn | Encoding, scaling, model training, `GridSearchCV` |
| Matplotlib / Seaborn | EDA visualizations |
| Jupyter Notebook | Development environment |

---

## 📝 Future Improvements

- Re-run all three models with `OneHotEncoder` applied to `Marital_Status` and `Gender` instead of `LabelEncoder`, to test sensitivity of Naive Bayes to nominal encoding choice
- Add 5-fold cross-validation for all three models to confirm stability beyond a single train/test split
- Try `RandomForestClassifier` and `XGBoost` as stronger baselines — both naturally handle the categorical/imbalance issues that affected Naive Bayes
- Build a single comparison table script that re-trains all models with identical preprocessing for a fully apples-to-apples benchmark

---

## 👤 Author

**Vikas Ajay Vishwakarma**
BSc IT Student | Aspiring Data Scientist  
🔗 [GitHub](https://github.com/codewithvikas96-ui)
