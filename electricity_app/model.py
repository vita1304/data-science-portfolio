import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump


def load_clean_data(csv_path: str = "procurement_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def train_polynomial_ridge_model(X: pd.DataFrame, y: pd.Series):
    candidate_alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        RidgeCV(alphas=candidate_alphas)
    )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    # Created csv from my own report, so I can use it for training and testing, later I'll try to add more months to the dataset and retrain the model
    data = load_clean_data("procurement_clean.csv")
    print(f"Дані успішно завантажено. Доступно місяців для аналізу: {len(data)}")

    features = ["Gas_EUR_MWh", "Diesel_UAH_l"]
    X = data[features]
    y = data["Electricity_EUR_MWh"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly_ridge_model = train_polynomial_ridge_model(X_train, y_train)
    predictions = poly_ridge_model.predict(X_test)

    best_alpha = poly_ridge_model.named_steps["ridgecv"].alpha_
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Підібране alpha (RidgeCV): {best_alpha}")
    print(f"R² на тесті: {r2:.3f}")
    print(f"Похибка RMSE: {rmse:.2f} EUR / MWh")

    final_model = train_polynomial_ridge_model(X, y)
    dump(final_model, "electricity_poly_ridge_model.joblib")
    dump(data, "clean_energy_data.joblib")
    print("Модель та дані успішно збережено.")