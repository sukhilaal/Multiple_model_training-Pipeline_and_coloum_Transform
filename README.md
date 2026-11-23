# Multiple_model_training-Pipeline_and_coloum_Transform
End-to-end ML classification pipeline in Python predicting meal time (Lunch vs. Dinner) on the Seaborn Tips dataset. Features automated Scikit-Learn ColumnTransformer preprocessing, multi-model benchmarking (Logistic Regression, SVC, Decision Tree), Random Forest RandomizedSearchCV tuning, and OOB evaluation
# Restaurant Meal-Time Prediction & Automated ML Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipelines-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Dataset%20%26%20Viz-4C72B0)

An end-to-end supervised machine learning project built in Python and Scikit-Learn that predicts whether a restaurant visit occurred during **Lunch** or **Dinner** using the Seaborn `tips` dataset. The notebook demonstrates automated feature engineering with `Pipeline` and `ColumnTransformer`, multi-model classification benchmarking, `RandomizedSearchCV` hyperparameter optimization, and Out-of-Bag (`OOB`) ensemble validation.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Overview](#-dataset-overview)
- [Automated Preprocessing Pipeline](#-automated-preprocessing-pipeline)
- [Model Benchmarking & Results](#-model-benchmarking--results)
- [Random Forest Tuning & OOB Evaluation](#-random-forest-tuning--oob-evaluation)
- [Getting Started](#-getting-started)

---

## 🎯 Project Overview

Manual preprocessing steps often introduce data leakage and repetitive boilerplate code. This notebook structures a clean, reproducible workflow that:
1. **Explores and encodes** target classes (`time`: Lunch vs. Dinner) using `LabelEncoder`.
2. **Automates feature engineering** across mixed numerical and categorical data types using `Pipeline` and `ColumnTransformer`.
3. **Benchmarks multiple classifiers** (`LogisticRegression`, `SVC`, and `DecisionTreeClassifier`) via a reusable evaluation function (`model_train_eval`).
4. **Optimizes a Random Forest Classifier** using 5-fold `RandomizedSearchCV`.
5. **Demonstrates Out-of-Bag (`OOB`) scoring** on a synthetic 20-feature binary classification dataset.

---

## 📊 Dataset Overview

The primary workflow uses Seaborn's built-in **`tips`** dataset ($244$ records, $7$ columns) with an $80/20$ train-test split ($195$ training samples, $49$ test samples, `random_state=42`):

| Feature | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `total_bill` | Numerical (`float64`) | Input ($X$) | Total restaurant bill amount (USD) |
| `tip` | Numerical (`float64`) | Input ($X$) | Tip amount given (USD) |
| `size` | Numerical (`int64`) | Input ($X$) | Number of diners in the party ($1$ to $6$) |
| `sex` | Categorical (`category`) | Input ($X$) | Gender of the bill payer (`Male`, `Female`) |
| `smoker` | Categorical (`category`) | Input ($X$) | Whether the party included smokers (`Yes`, `No`) |
| `day` | Categorical (`category`) | Input ($X$) | Day of the week (`Thur`, `Fri`, `Sat`, `Sun`) |
| `time` | Binary (`int64`, encoded) | Target ($y$) | Meal time (`0` = Dinner, `1` = Lunch) |

---

## ⚙️ Automated Preprocessing Pipeline

To prevent data leakage, transformations are fit exclusively on `x_train` and applied to `x_test` using Scikit-Learn's `ColumnTransformer`, expanding the input feature space from $6$ raw columns to $11$ processed features:

- **Numerical Pipeline (`total_bill`, `tip`, `size`):**
  1. `SimpleImputer(strategy='median')` — handles missing numerical entries.
  2. `StandardScaler()` — standardizes features to zero mean and unit variance ($\mu = 0, \sigma = 1$).
- **Categorical Pipeline (`sex`, `smoker`, `day`):**
  1. `SimpleImputer(strategy='most_frequent')` — imputes missing categorical values using the mode.
  2. `OneHotEncoder(handle_unknown='ignore')` — encodes categories into binary indicator columns.

---

## 📈 Model Benchmarking & Results

Three baseline classification algorithms were trained and evaluated on the $49$-sample test set:

| Model | Test Accuracy | Class `0` F1-Score | Class `1` F1-Score | Macro Avg F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** (`liblinear`) | **100.00%** | 1.00 | 1.00 | 1.00 |
| **Support Vector Classifier (`SVC`)** | **95.92%** | 0.97 | 0.93 | 0.95 |
| **Decision Tree Classifier** | **93.88%** | 0.96 | 0.90 | 0.93 |

*Note: Logistic Regression achieved a perfect confusion matrix (`[[34, 0], [0, 15]]`) on the test split due to strong day-of-week and dining-time separation in the `tips` dataset.*

---

## 🌲 Random Forest Tuning & OOB Evaluation

### 1. Hyperparameter Optimization (`RandomizedSearchCV`)
A `RandomForestClassifier` was tuned over $10$ random parameter candidates using $5$-fold cross-validation ($50$ total fits):
- **Search Space:**
  - `n_estimators`: `[30, 40, 50, 100]`
  - `max_depth`: `[1, 2, 3, 4, 5, 10, None]`
  - `criterion`: `['gini', 'entropy']`
- **Best Hyperparameters:** `{'n_estimators': 30, 'max_depth': 5, 'criterion': 'entropy'}`
- **Best 5-Fold CV Accuracy:** **`95.90%`** ($0.95897$)

### 2. Out-of-Bag (`OOB`) Validation Experiment
To demonstrate internal ensemble validation without a separate validation split, a synthetic binary classification dataset was generated using `make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=1)`:
- **Model:** `RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1, n_jobs=-1)`
- **Out-of-Bag (`OOB`) Score:** **`86.10%`** ($0.8610$)

---

## 🚀 Getting Started

### Prerequisites
Install the required Python libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook
1. Clone this repository or download the `.ipynb` notebook file.
2. Launch Jupyter Notebook or Google Colab:
   ```bash
   jupyter notebook
   ```
3. Run all cells sequentially to reproduce the preprocessing pipeline, model evaluations, and hyperparameter search results.
