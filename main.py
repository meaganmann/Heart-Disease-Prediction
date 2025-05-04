import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    brier_score_loss, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("processed.cleveland.data", header=None, na_values='?')
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Drop rows with missing values
df.dropna(inplace=True)

# Binary target: 0 = no disease, 1 = disease
df['target'] = (df['target'] > 0).astype(int)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=["cp", "restecg", "slope", "thal"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### 1. Logistic Regression (Sklearn)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_proba_lr = clf.predict_proba(X_test)[:, 1]

print("Logistic Regression Report")
print(classification_report(y_test, y_proba_lr > 0.5))
print("ROC AUC:", roc_auc_score(y_test, y_proba_lr))

### 2. Bayesian Logistic Regression (Pyro)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)

X_test_tensor = torch.tensor(X_test, dtype=torch.float)

# Define model
def model(X, y=None):
    w_prior = dist.Normal(torch.zeros(X.shape[1]), torch.ones(X.shape[1])).to_event(1)
    b_prior = dist.Normal(0., 1.)
    
    w = pyro.sample("weights", w_prior)
    b = pyro.sample("bias", b_prior)
    
    logits = X.matmul(w) + b
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

# Define guide
def guide(X, y=None):
    w_loc = pyro.param("w_loc", torch.zeros(X.shape[1]))
    w_scale = pyro.param("w_scale", torch.ones(X.shape[1]), constraint=dist.constraints.positive)
    b_loc = pyro.param("b_loc", torch.tensor(0.))
    b_scale = pyro.param("b_scale", torch.tensor(1.), constraint=dist.constraints.positive)
    
    pyro.sample("weights", dist.Normal(w_loc, w_scale).to_event(1))
    pyro.sample("bias", dist.Normal(b_loc, b_scale))

# Setup SVI
pyro.clear_param_store()
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Training loop
num_steps = 5000
for step in range(num_steps):
    loss = svi.step(X_train_tensor, y_train_tensor)
    if step % 500 == 0:
        print(f"Step {step} : loss = {loss}")

# Draw samples for prediction
num_samples = 100
predictive_probs = []

for _ in range(num_samples):
    sampled_w = dist.Normal(
        pyro.param("w_loc"), pyro.param("w_scale")
    ).sample()
    sampled_b = dist.Normal(
        pyro.param("b_loc"), pyro.param("b_scale")
    ).sample()

    logits = X_test_tensor @ sampled_w + sampled_b
    probs = torch.sigmoid(logits)
    predict_probs = probs.detach().numpy()
    predictive_probs.append(predict_probs)

# Aggregate predictions
predictive_probs = np.array(predictive_probs)
mean_probs = predictive_probs.mean(axis=0)
entropy = -np.mean(predictive_probs * np.log(predictive_probs + 1e-8), axis=0)

# Evaluate Bayesian model
print("Bayesian Logistic Regression Report")
print(classification_report(y_test, mean_probs > 0.5))
print("ROC AUC:", roc_auc_score(y_test, mean_probs))
print("Brier Score:", brier_score_loss(y_test, mean_probs))

### 3. Plotting

# ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_bayes, tpr_bayes, _ = roc_curve(y_test, mean_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_bayes, tpr_bayes, label="Bayesian Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Uncertainty histogram
plt.figure(figsize=(8, 6))
sns.histplot(entropy, kde=True)
plt.title("Predictive Entropy of Bayesian Model")
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
