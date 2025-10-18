# CompareMLModels.py

"""
File name: CompareMLModels.py
Author: Naran Bat
Description: Compare various ML models on the Iris dataset
Inputs: iris.csv (Iris dataset)
Outputs: 
    - Confusion matrix
    - Overall accuracy
Date created: 2025-09-08
""" 
# Chatgpt helped to get the necessary imports
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline


# load dataset 
df = pd.read_csv("iris.csv", sep=",", header=None)

X = df.iloc[:, :-1].values   # features of iris dataset
y = df.iloc[:, -1].values    # target labels


# encode class labels 
encoder = preprocessing.LabelEncoder() # init encoder
y_encoded = encoder.fit_transform(y) # fit and transform labels
class_names = encoder.classes_ # class names

# 2-fold cross validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)

# machine learning models to compare
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Polynomial Regression (deg=2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Polynomial Regression (deg=3)": make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    "Naive Bayes": GaussianNB(),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
}

def run_model(name, model):
    all_preds = []
    all_true = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx] # features
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx] # labels


        model.fit(X_train, y_train)

        # regression models for classification
        if "Regression" in name:
            y_pred = model.predict(X_test)
            y_pred = np.rint(y_pred).astype(int)  # round to nearest integer
            y_pred = np.clip(y_pred, 0, 2)  # clip to valid class range
        # classification models
        else:
            y_pred = model.predict(X_test) # predict classes

        all_preds.extend(y_pred) # collect all predictions
        all_true.extend(y_test) # collect all true labels

    cm = confusion_matrix(all_true, all_preds) 
    acc = accuracy_score(all_true, all_preds)

    print(f"\nModel: {name}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")

# run and compare all models
for name, model in models.items():
    run_model(name, model)
