import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump


def load_clean_data(csv_path: str = "procurement_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Month_Num" not in df.columns and "Month" in df.columns:
        df["Month_Num"] = pd.to_datetime(df["Month"]).dt.month
    return df


def train_polynomial_ridge_model(X: pd.DataFrame, y: pd.Series):
    # Починаємо від 5.0, щоб придушити паразитарні від'ємні параболи
    candidate_alphas = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    
    regressor = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=candidate_alphas)
    )
    
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log,
        inverse_func=np.exp
    )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    data = load_clean_data("procurement_clean.csv")
    print(f"Дані успішно завантажено. Доступно місяців для аналізу: {len(data)}")

    features = ["Gas_EUR_MWh", "Diesel_UAH_l", "Month_Num", "EUR_UAH"]
    X = data[features]
    y = data["Electricity_EUR_MWh"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly_ridge_model = train_polynomial_ridge_model(X_train, y_train)
    predictions = poly_ridge_model.predict(X_test)

    best_alpha = poly_ridge_model.regressor_.named_steps["ridgecv"].alpha_
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Підібране alpha (RidgeCV): {best_alpha}")
    print(f"R² на тесті: {r2:.3f}")
    print(f"Похибка RMSE: {rmse:.2f} EUR / MWh")

    final_model = train_polynomial_ridge_model(X, y)
    dump(final_model, "electricity_poly_ridge_model.joblib")
    dump(data, "clean_energy_data.joblib")
    print("Модель та дані успішно збережено.")
