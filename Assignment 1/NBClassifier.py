"""
File name: NBClassifier.py
Author: Naran Bat
Description: Naive Bayes Classifier for Iris Dataset
Inputs: iris.csv (Iris dataset)
Outputs: 
    - Overall accuracy
    - Confusion matrix
    - Classification report (precision, recall, f1-score)
Date created: 2025-08-26
""" 
# chatgpt helped to get the necessary imports
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load dataset 
df = pd.read_csv("iris.csv", sep=",", header=None)

X = df.iloc[:, :-1].values   # features of iris dataset
y = df.iloc[:, -1].values    # target labels

# encode class labels 
encoder = preprocessing.LabelEncoder() # init encoder
y_encoded = encoder.fit_transform(y) # fit and transform labels
class_names = encoder.classes_ # class names

# 2-fold cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=42) 
model = GaussianNB() # Naive Bayes classifier

# storage for predicted labels
y_pred = np.zeros_like(y_encoded)

for train_idx, test_idx in kf.split(X):
    model.fit(X[train_idx], y_encoded[train_idx]) # train model
    y_pred[test_idx] = model.predict(X[test_idx]) # predict labels

# compute metrics
accuracy = accuracy_score(y_encoded, y_pred)
print("Overall Accuracy:", round(accuracy, 3))

cm = confusion_matrix(y_encoded, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_encoded, y_pred, target_names=class_names))
