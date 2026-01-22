# Heart Disease Prediction with Logistic and Bayesian Models

## Overview
This project compares **standard logistic regression** and **Bayesian logistic regression** for predicting the presence of heart disease using the **UCI Cleveland Heart Disease dataset**. The goal is to evaluate predictive performance while highlighting the benefits of Bayesian methods for modeling uncertainty.

---

## Objective
- Perform **binary classification** to predict heart disease
- Compare frequentist and Bayesian modeling approaches
- Analyze predictive uncertainty in model outputs

---

## Models Implemented
- **Logistic Regression** (scikit-learn)
- **Bayesian Logistic Regression** (Pyro)

---

## Dataset
- **Source:** UCI Cleveland Heart Disease Dataset  
- **File:** `processed.cleveland.data`  
- **Preprocessing:** Dataset is preprocessed prior to model training

---

## Evaluation Metrics
- Classification accuracy  
- ROC AUC  
- Brier score  
- ROC curve comparison  
- Predictive uncertainty analysis using entropy histograms  

---

## Output
The project generates:
- Performance metrics for both models
- ROC curves comparing classification performance
- Visualizations illustrating predictive uncertainty in the Bayesian model

---

## How to Run

### Install Dependencies
```bash
pip install pandas numpy torch pyro-ppl scikit-learn matplotlib seaborn
