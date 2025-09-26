import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

MODEL_PATH = "model_xgboost_classifier.joblib"

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["grav"])
    X = df.drop(columns=["grav"])
    y = df["grav"]
    y = df["grav"].astype(int) - 1
    return X, y

def train_model(X_train, y_train):
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=4,  # car classes = 1.0, 2.0, 3.0, 4.0
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    dump(model, MODEL_PATH)
    print(f"Modèle XGBoost sauvegardé dans {MODEL_PATH}")
    return model

def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        print("Modèle trouvé, chargement...")
        return load(MODEL_PATH)
    else:
        return train_model(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nÉvaluation du modèle XGBoost :")
    print(classification_report(y_test + 1, y_pred + 1))

def main():
    X, y = load_data("../data/accidents_2021-2023_clean.csv")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    model = load_or_train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
