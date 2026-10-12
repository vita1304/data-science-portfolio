import os
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

def load_model_and_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, "electricity_poly_ridge_model.joblib")
    data_path = os.path.join(current_dir, "clean_energy_data.joblib")

    model = load(model_path)
    df = load(data_path)
    return model, df

def create_streamlit_app():
    st.set_page_config(
        page_title="Прогноз вартості електроенергії",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Прогноз вартості електроенергії ")
    st.markdown(
        """
        Додаток моделює динаміку ринку електроенергії на базі цін природного газу 
        (конвертованого в європейські енергетичні одиниці **EUR / MWh**) та вартості палива.
        """
    )

    try:
        model, df = load_model_and_data()
    except FileNotFoundError:
        st.error("Файли моделі або даних не знайдені. Спершу запустити `python model.py`!")
        return

    st.sidebar.header("Вхідні параметри факторів")

    gas_eur_val = st.sidebar.slider(
        "Ціна газу (EUR / MWh)",
        min_value=10.0,
        max_value=80.0,  
        value=50.0,           
        step=0.5
    )

    diesel_val = st.sidebar.slider(
        "Ціна ДП (грн / л)",
        min_value=30.0,
        max_value=100.0,
        value=60.0,          
        step=0.5
    )
    
    month_val = st.sidebar.slider(
        "Номер месяца (1-12)",
        min_value=1, 
        max_value=12, 
        value=6
    )
    
    eur_uah_val = st.sidebar.number_input(
        "Курс EUR/UAH", 
        value=42.0, 
        step=0.1
    )

    if st.sidebar.button("Розрахувати прогноз", type="primary"):
        input_data = pd.DataFrame([{
            "Gas_EUR_MWh": gas_eur_val,
            "Diesel_UAH_l": diesel_val,
            "Month_Num": month_val,
            "EUR_UAH": eur_uah_val  
        }])

        pred_mwh = float(model.predict(input_data)[0])

        current_eur_rate = float(df["EUR_UAH"].iloc[-1])
        pred_uah_kw = (pred_mwh * current_eur_rate) / 1000

        col1, col2, col3 = st.columns(3)
        col1.metric("Очікувана ціна е/е", f"{pred_mwh:.2f} € / MWh")
        col2.metric("Еквівалент у грн", f"{pred_uah_kw:.2f} грн / кВт·год")
        col3.metric("Базовий курс євро", f"{current_eur_rate:.2f} грн/€")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            df["Gas_EUR_MWh"],
            df["Electricity_EUR_MWh"],
            color="gray",
            alpha=0.6,
            label="Історичні місяці (факт)"
        )

        gas_range = np.linspace(df["Gas_EUR_MWh"].min(), df["Gas_EUR_MWh"].max() * 1.2, 150)
        
        # Виправлений блок для поліноміальної кривої, який викликав помилку
        curve_input = pd.DataFrame({
            "Gas_EUR_MWh": gas_range,
            "Diesel_UAH_l": np.full_like(gas_range, diesel_val),
            "Month_Num": np.full_like(gas_range, month_val),
            "EUR_UAH": np.full_like(gas_range, eur_uah_val)
        })
        predicted_curve = model.predict(curve_input)

        ax.plot(
            gas_range,
            predicted_curve,
            color="green",
            linewidth=2,
            label=f"Поліноміальна крива (при ДП = {diesel_val:.1f} грн)"
        )

        ax.scatter(
            [gas_eur_val],
            [pred_mwh],
            color="red",
            s=120,
            zorder=5,
            label=f"Ваш прогноз: {pred_mwh:.1f} €/MWh"
        )

        ax.axvline(gas_eur_val, color="red", linestyle=":", alpha=0.5)
        ax.axhline(pred_mwh, color="red", linestyle=":", alpha=0.5)

        ax.set_title("Модель ціноутворення: Газ vs Електроенергія (EUR / MWh)", fontsize=13)
        ax.set_xlabel("Ціна природного газу (EUR / MWh)", fontsize=11)
        ax.set_ylabel("Ціна електроенергії (EUR / MWh)", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)
    else:
        st.info("Виберіть параметри на бічній панелі ліворуч та натисніть **'Розрахувати прогноз'**.")

if __name__ == "__main__":
    create_streamlit_app()
