# 💳 Credit Card Fraud Detection

> Detecting fraudulent bank transactions with machine learning, on a real-world, highly imbalanced dataset.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
</p>

---

## 📑 Table of Contents

- [Context](#-context)
- [The Dataset](#-the-dataset)
- [The Challenge: an Imbalanced Dataset](#-the-challenge-an-imbalanced-dataset)
- [Approach](#-approach)
- [Results](#-results)
- [Conclusion](#-conclusion)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

---

## 🎯 Context

Credit card fraud is a major challenge for financial institutions. The goal of this project is to **build and compare several Machine Learning models** able to correctly classify a transaction as **legitimate** or **fraudulent**.

The main difficulty is not so much the choice of model, but **handling the extreme class imbalance**: fraudulent transactions account for less than 0.2 % of the data.

---

## 📊 The Dataset

The dataset contains transactions made by European cardholders over two days in September 2013.

| Property | Value |
|---|---|
| Total transactions | **284,807** |
| Fraudulent transactions | **492** (≈ 0.172 %) |
| Number of features | 31 |

**Feature description:**

- `V1` … `V28`: features **anonymized via PCA** (bank data confidentiality).
- `Time`: seconds elapsed since the first transaction.
- `Amount`: transaction amount.
- `Class`: target variable — `0` = legitimate, `1` = fraud.

📥 Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## ⚠️ The Challenge: an Imbalanced Dataset

With only **0.17 % of frauds**, a model that always predicts "everything is legitimate" would already reach **99.8 % accuracy**… while being completely useless.

That's why we don't rely on *accuracy*, but rather on **recall** (ability to catch real frauds), **precision** (avoiding false alarms), and **AUPRC** (Area Under the Precision-Recall Curve).

---

## 🔧 Approach

The notebook follows an 8-step workflow:

1. **Data exploration** — feature distributions, missing-value checks, measuring the imbalance.
2. **Preprocessing** — scaling `Amount` with a `RobustScaler` (robust to outliers) and normalizing `Time` to the [0, 1] range.
3. **Data preparation** — splitting into train / test / validation sets.
4. **Initial evaluation** — 4 models trained on the **imbalanced** data.
5. **Rebalancing** — building a balanced dataset (equal number of frauds and non-frauds) via undersampling.
6. **Second evaluation** — the same models, re-evaluated on the **balanced** data.
7. **Optimization** — hyperparameter tuning of the best model with `GridSearchCV`.
8. **Conclusion**.

**Models compared:** Logistic Regression · Random Forest · Gradient Boosting · Linear SVC.

---

## 📈 Results

### Before rebalancing (recall on the "Fraud" class)

On imbalanced data, all models **miss a large share of frauds**:

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|---|:---:|:---:|:---:|
| Logistic Regression | 0.83 | 0.28 | 0.42 |
| Random Forest | 0.89 | 0.44 | 0.59 |
| Gradient Boosting | 0.71 | 0.56 | 0.62 |
| Linear SVC | 1.00 | 0.44 | 0.62 |

➡️ Even the best models barely catch half of the frauds.

### After rebalancing

Rebalancing **dramatically improves** detection:

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | AUPRC |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **1.00** | **0.91** | **0.96** | **0.9900** |
| Linear SVC | 0.98 | 0.93 | 0.96 | — |
| Random Forest | 1.00 | 0.87 | 0.93 | — |
| Gradient Boosting | 0.94 | 0.93 | 0.94 | — |

➡️ **Logistic Regression** comes out as the best model (AUPRC = **0.9900**).

### After optimization

After `GridSearchCV` (optimizing for AUPRC), the best hyperparameters are:

```python
{'C': 1, 'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}
```

Final AUPRC: **0.9834** — slightly lower than the initial score, but the model is **more robust and generalizes better** to unseen data.

---

## ✅ Conclusion

- **Class imbalance** is the real heart of the problem: without handling it, even good models miss most of the frauds.
- After rebalancing, **Logistic Regression** offers the best trade-off between performance and simplicity.
- A suitable metric (**AUPRC**) is essential to correctly evaluate this kind of problem.

---

## 🚀 Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/axelcohen75/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn jupyter

# 3. Download the dataset from Kaggle and place it at the root
#    (file: creditcard.csv)

# 4. Launch the notebook
jupyter notebook Credit-Card-Fraud-Detection.ipynb
```

> ℹ️ The `creditcard.csv` file (~150 MB) is not included in the repository. Download it directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
├── Credit-Card-Fraud-Detection.ipynb   # Main notebook (analysis + models)
├── creditcard.csv                      # Data (to download from Kaggle)
└── README.md                           # This file
```

---

## 🔮 Future Improvements

- Try more advanced rebalancing techniques (**SMOTE**, oversampling) instead of plain undersampling, which discards a lot of data.
- Experiment with more sophisticated models (**XGBoost**, **LightGBM**, neural networks).
- Set up stratified cross-validation on the full dataset.
- Tune the **decision threshold** based on the business cost of a false negative vs. a false positive.

---

<p align="center"><i>Project built as part of learning Machine Learning.</i></p>
