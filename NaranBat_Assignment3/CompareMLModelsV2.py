# CompareMLModelsV2.py

# Chatgpt helped to get the necessary imports
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")  # silence convergence warnings for demo

def lr_as_classifier(X, y, cv, poly_degree=None):
    # create a regression model pipeline
    if poly_degree is None:
        # linear regression
        model = make_pipeline(StandardScaler(), LinearRegression())
    else:
        # polynomial regression
        model = make_pipeline(StandardScaler(),
                              PolynomialFeatures(degree=poly_degree, include_bias=False),
                              LinearRegression())
    # cross_val_predict returns continuous predictions for regressor; we round after
    preds_cont = cross_val_predict(model, X, y, cv=cv) 
    preds_rounded = np.rint(preds_cont).astype(int)  # round to nearest integer
    # clip to valid labels range
    valid_labels = np.unique(y) 
    preds_clipped = np.clip(preds_rounded, valid_labels.min(), valid_labels.max()) 
    return preds_clipped

def classifier_predict(clf, X, y, cv):
    # Use cross_val_predict to get predictions from classifier
    return cross_val_predict(clf, X, y, cv=cv)

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    labels = iris.target_names

    # 2-fold stratified CV so each fold preserves class ratios
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    # Define models to evaluate
    models = [
        ("Naive Bayes (GaussianNB)", GaussianNB()),
        ("Linear Regression (LR -> round)", "LR_REG"),  # special handling
        ("Polynomial deg2 Regression (LR poly2 -> round)", ("LR_POLY", 2)),
        ("Polynomial deg3 Regression (LR poly3 -> round)", ("LR_POLY", 3)),
        ("k-NN (KNeighborsClassifier)", KNeighborsClassifier(n_neighbors=5)),
        ("LDA (LinearDiscriminantAnalysis)", LDA()),
        ("QDA (QuadraticDiscriminantAnalysis)", QDA()),
        ("SVM (LinearSVC)", make_pipeline(StandardScaler(), svm.LinearSVC(max_iter=20000, dual=False, random_state=42))),
        ("Decision Tree (DecisionTreeClassifier)", DecisionTreeClassifier(random_state=42)),
        ("Random Forest (RandomForestClassifier)", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("ExtraTrees (ExtraTreesClassifier)", ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ("Neural Network (MLPClassifier)", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)))
    ]
    # Evaluate each model
    print("CompareMLModelsV2: 2-fold cross-validated predictions on Iris (150 samples)\n")
    for name, clf in models:
        print("=== Model:", name, "===")
        if isinstance(clf, str) or (isinstance(clf, tuple) and clf[0] == "LR_POLY"):
            # handle regression-as-classifier options
            if clf == "LR_REG" or name.startswith("Linear Regression"):
                y_pred = lr_as_classifier(X, y, cv=cv, poly_degree=None) # linear regression
            elif isinstance(clf, tuple) and clf[0] == "LR_POLY":
                degree = clf[1]
                y_pred = lr_as_classifier(X, y, cv=cv, poly_degree=degree) # polynomial regression
            else:
                # fallback
                y_pred = lr_as_classifier(X, y, cv=cv, poly_degree=None) # linear regression
        else:
            # normal classifier: use cross_val_predict with cv
            y_pred = classifier_predict(clf, X, y, cv=cv)

        # Compute confusion matrix and accuracy
        cm = confusion_matrix(y, y_pred, labels=[0,1,2])
        acc = accuracy_score(y, y_pred)
        print("Confusion Matrix (rows: true classes 0,1,2 ; cols: predicted 0,1,2):")
        print(cm)
        print("Sum of confusion matrix entries (should be 150):", cm.sum())
        if cm.sum() != 150:
            print("WARNING: confusion matrix does not sum to 150. Check CV or labels.")
        print("Accuracy: {:.4f}".format(acc))
        print()  # blank line


if __name__ == "__main__":
    main()
