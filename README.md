# Heart-Disease-Prediction

This project compares standard logistic regression and Bayesian logistic regression (via Pyro) for predicting heart disease using the UCI Cleveland Heart Disease dataset.

Overview
Goal: Predict presence of heart disease (binary classification)

Models:
Logistic Regression (scikit-learn)
Bayesian Logistic Regression (Pyro)

Dataset: processed.cleveland.data

Output: Classification metrics (accuracy, ROC AUC, Brier score)

ROC curve comparison: Predictive uncertainty (entropy histogram)

How to Run
Install dependencies:
pip install pandas numpy torch pyro-ppl scikit-learn matplotlib seaborn
Download the dataset and place processed.cleveland.data in the same folder.

Run the script:
python heart_disease_prediction.py
