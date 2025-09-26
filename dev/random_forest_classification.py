import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from joblib import dump

MODEL_PATH = "../data/models/model_random_forest.joblib"

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["grav"])
    X = df.drop(columns=["grav"])
    y = df["grav"]
    return X, y

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    dump(clf, MODEL_PATH)
    print(f"Modèle entraîné et sauvegardé dans {MODEL_PATH}")
    return clf

def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        print("Modèle trouvé, chargement...")
        return load(MODEL_PATH)
    else:
        return train_model(X_train, y_train)

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("\nÉvaluation du modèle :")
    print(classification_report(y_test, y_pred))

def plot_feature_importances(clf, feature_names, top_n=20):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title("Importance des variables (top 20)")
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data("../data/accidents_2021-2023_clean.csv")

    # Imputation simple au cas où certaines colonnes ont des NaN
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    clf = load_or_train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    plot_feature_importances(clf, X.columns)
    dump(clf, 'random_forest_classification.joblib')


if __name__ == "__main__":
    main()
