# 🎓 Student Dropout Prediction

Predicting which students are at risk of dropping out using academic, demographic, and socio-economic data — built on the [UCI "Predict Students' Dropout and Academic Success"](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset.

## 📌 Overview

Early identification of at-risk students lets institutions intervene before it's too late. This project frames dropout prediction as a **binary classification problem** (`Dropout` vs. `Graduate`/`Enrolled`) and walks through the full pipeline — from raw UCI data to a saved, deployable model.

## 🗂️ Dataset

- **Source:** UCI Machine Learning Repository (ID: 697), pulled directly via `ucimlrepo`
- **Records:** ~4,400 students
- **Features:** Demographics, socio-economic indicators (unemployment rate, inflation, GDP), and academic performance (curricular units enrolled/approved/graded per semester)
- **Target:** Collapsed from 3 classes (`Dropout`, `Enrolled`, `Graduate`) into a binary label — `1 = Dropout`, `0 = Not Dropout`

## 🔧 Pipeline

1. **EDA** — nulls, duplicates, correlation heatmap of academic features
2. **Preprocessing** — one-hot encoding for categoricals, `StandardScaler` for numerics via `ColumnTransformer`
3. **Modeling** — compared multiple approaches:
   - Logistic Regression (baseline, class-weighted)
   - Random Forest (default + `GridSearchCV`-tuned)
   - Final Logistic Regression pipeline (`StandardScaler` + `LogisticRegression`)
4. **Threshold tuning** — swept decision thresholds (0.10–0.90) on a validation set to maximize F1, since the classes are imbalanced and the default 0.5 cutoff under-catches dropouts
5. **Evaluation** — accuracy, precision/recall, F1, confusion matrix, and ROC-AUC on a held-out test set
6. **Persistence** — final model and chosen threshold saved with `joblib` (`churn_model.pkl`, `churn_threshold.pkl`) for reuse

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~0.88 |
| Precision (Dropout) | ~0.84 |
| Recall (Dropout) | ~0.77–0.82 |
| F1 (Dropout) | ~0.82 |
| ROC-AUC | ~0.93 |

*(Exact numbers depend on the tuned threshold — see notebook output.)*

## 🛠️ Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, `GridSearchCV`, `Pipeline`)
- seaborn / matplotlib for visualization
- `ucimlrepo` for data access, `joblib` for model persistence

## 🚀 Getting Started

```bash
pip install pandas numpy scikit-learn seaborn matplotlib ucimlrepo joblib
jupyter notebook student_dropout.ipynb
```

## 📁 Repo Structure

```
├── student_dropout.ipynb   # full analysis + modeling notebook
├── churn_model.pkl         # trained final model (generated on run)
├── churn_threshold.pkl     # tuned decision threshold (generated on run)
└── README.md
```

## 💡 Key Takeaways

- Academic performance features (curricular units approved/grades, especially 1st and 2nd semester) are the strongest predictors of dropout.
- Default 0.5 classification thresholds aren't optimal for imbalanced targets — tuning the threshold against F1 materially improves recall on the minority (dropout) class.
- A simple, well-tuned Logistic Regression pipeline performed competitively with Random Forest, while staying more interpretable and faster to deploy.

## 📄 License

This project is open for educational and research use. Dataset © original UCI contributors.
