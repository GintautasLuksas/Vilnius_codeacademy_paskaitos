import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calculate_mse_with_and_without_features():
    # Gauti duomenų rinkinį
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    # Padalinti duomenis į mokymo ir testavimo rinkinius
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sukurti ir apmokyti linijinės regresijos modelį su visais kintamaisiais
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prognozuoti naudojant testavimo duomenis
    y_pred_all = model.predict(X_test)

    # Apskaičiuoti MSE su visais kintamaisiais
    mse_all_features = mean_squared_error(y_test, y_pred_all)

    # Saugojame rezultatus
    mse_values = {'All Features': mse_all_features}

    # Iteruoti per kintamuosius ir po vieną pašalinti
    for feature in X.columns:
        X_train_reduced = X_train.drop(columns=[feature])
        X_test_reduced = X_test.drop(columns=[feature])

        # Sukurti ir apmokyti modelį be vieno kintamojo
        model.fit(X_train_reduced, y_train)
        y_pred_reduced = model.predict(X_test_reduced)

        # Apskaičiuoti MSE be vieno kintamojo
        mse_reduced = mean_squared_error(y_test, y_pred_reduced)
        mse_values[f'Without {feature}'] = mse_reduced

    # Pavaizduoti MSE vertes diagramoje
    plt.figure(figsize=(10, 6))
    plt.barh(list(mse_values.keys()), list(mse_values.values()), color='skyblue')
    plt.xlabel('MSE')
    plt.ylabel('Kintamieji')
    plt.title('MSE reikšmės, pašalinus po vieną kintamąjį')
    plt.show()

if __name__ == "__main__":
    calculate_mse_with_and_without_features()
