import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

MODEL_PATH = "../data/models/model_random_forest_reg.joblib"

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["grav"])
    X = df.drop(columns=["grav"])
    y = df["grav"]
    return X, y

def train_model(X_train, y_train):
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    dump(reg, MODEL_PATH)
    print(f"Modèle de régression sauvegardé dans {MODEL_PATH}")
    return reg

def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        print("Modèle trouvé, chargement...")
        return load(MODEL_PATH)
    else:
        return train_model(X_train, y_train)

def evaluate_model(reg, X_test, y_test):
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nÉvaluation du modèle de régression :")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R² score: {r2:.3f}")

    # Affichage de la comparaison vraies vs prédites
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([1, 4], [1, 4], 'r--')
    plt.xlabel("Gravité réelle")
    plt.ylabel("Gravité prédite")
    plt.title("Régression : gravité réelle vs prédite")
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data("../data/accidents_2021-2023_clean.csv")
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    reg = load_or_train_model(X_train, y_train)
    evaluate_model(reg, X_test, y_test)
    dump(reg, 'random_forest_regression.joblib')

if __name__ == "__main__":
    main()
