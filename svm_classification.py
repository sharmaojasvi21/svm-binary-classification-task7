import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# Train RBF SVM
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Evaluation
print("Linear Kernel Classification Report:")
print(classification_report(y_test, y_pred_linear))
print("Confusion Matrix:
", confusion_matrix(y_test, y_pred_linear))

print("\nRBF Kernel Classification Report:")
print(classification_report(y_test, y_pred_rbf))
print("Confusion Matrix:
", confusion_matrix(y_test, y_pred_rbf))

# Hyperparameter Tuning
params = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
grid = GridSearchCV(SVC(kernel='rbf'), params, cv=5)
grid.fit(X_train, y_train)
print("\nBest Parameters from GridSearchCV:", grid.best_params_)

# Cross-validation
scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
print("Cross-validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
