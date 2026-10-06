# Heart Disease Prediction using Machine Learning

An end-to-end binary classification project that predicts whether a patient has heart disease from 13 clinical measurements.

## Problem
Given a patient's clinical parameters (age, sex, chest pain type, blood pressure, cholesterol, etc.), can we predict whether they have heart disease?

## Dataset
- Source: [Heart Disease UCI dataset on Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci/)
- 303 patients, 13 features + 1 target (`1` = disease, `0` = no disease)
- File: `data/heart-disease.csv`

## What this project covers
1. Exploratory data analysis (class balance, sex, age vs. max heart rate, chest pain type, correlation heatmap)
2. Training and comparing 3 models: Logistic Regression, K-Nearest Neighbors, Random Forest
3. Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`
4. Evaluation beyond accuracy: ROC curve and AUC, confusion matrix, precision, recall, F1
5. 5-fold cross-validation
6. Feature importance from Logistic Regression coefficients

## Results

| Item | Value |
|---|---|
| Best model | Logistic Regression (tuned) |
| Test accuracy | 88.5% |
| Cross-validated accuracy (5-fold) | 84.8% |
| Cross-validated precision / recall / F1 | 82.2% / 92.7% / 87.1% |
| AUC | 0.92 |

Other models on the same test split: Random Forest 83.6%, KNN 68.9% (75.4% after tuning `n_neighbors`).

> The goal set in the notebook was 95% accuracy. The tuned model did not reach it, so the notebook ends with ideas for the next experiments (more data, more models, further tuning).

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## How to run
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
jupyter notebook
```
Open `End-to-End Heart Disease Classification.ipynb` and choose **Kernel → Restart & Run All**. The notebook reads the data from `data/heart-disease.csv`, so keep that folder structure.

## Project structure
```
├── data/
│   └── heart-disease.csv
├── End-to-End Heart Disease Classification.ipynb
├── requirements.txt
└── README.md
```

## Acknowledgements
The project structure and explanations follow a widely used public tutorial for this dataset (Zero To Mastery's heart disease machine learning project by Daniel Bourke). The dataset comes from the UCI Machine Learning Repository via Kaggle.