# Task 7: Support Vector Machines (SVM) - AI & ML Internship

## Objective
Apply Support Vector Machines (SVMs) for both linear and non-linear classification using the Breast Cancer Wisconsin dataset.

## Tools Used
- Python
- scikit-learn
- NumPy
- Matplotlib

## What’s Covered
- Margin maximization
- Kernel trick (linear and RBF)
- Hyperparameter tuning (C, gamma)
- Cross-validation for evaluation

## Instructions
1. Load dataset using `sklearn.datasets.load_breast_cancer`
2. Preprocess with `StandardScaler`
3. Train `SVC` with linear and RBF kernel
4. Evaluate using accuracy, confusion matrix, classification report
5. Tune hyperparameters with `GridSearchCV`
6. Perform cross-validation with `cross_val_score`

## Results
- Linear and RBF SVMs compared
- GridSearchCV used to find optimal parameters
- Cross-validation ensured robust performance evaluation

## Dataset Used
Breast Cancer Dataset: Available in `scikit-learn`

## How to Run
```bash
python svm_classification.py
```
