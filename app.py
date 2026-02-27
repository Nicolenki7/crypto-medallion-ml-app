"""
Crypto Analytics & ML Predictor
================================
Medallion Architecture | Linear Regression | Interactive Dashboard

Author: Nicolas Zalazar
Date: 2026-02-28
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Crypto Predictive Analytics | Nicolas Zalazar",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Crypto ML: Predicción de Precios y Volatilidad")
st.markdown("---")

# 1. Carga de Datos (Independiente de Fabric)
@st.cache_data # Para que la app sea veloz
def load_data():
    df = pd.read_csv("gold_data.csv")
    df = df.dropna(subset=["Precio_Inicio_Mes", "Precio_Fin_Mes"])
    return df

try:
    df = load_data()
    
    # Sidebar para filtros
    st.sidebar.header("Configuración del Modelo")
    moneda = st.sidebar.selectbox("Seleccionar Activo", df['Nombre_Moneda'].unique())
    df_filtrado = df[df['Nombre_Moneda'] == moneda].copy()

    # 2. Replicación del Modelo ML
    X = df_filtrado[["Precio_Inicio_Mes", "Volatilidad_Media_Mensual", "Volumen_Promedio_Mensual"]].values
    y = df_filtrado["Precio_Fin_Mes"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Métricas de Performance
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # 3. Interfaz de Usuario: Predicción Interactiva
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulador de Predicción")
        p_inicio = st.number_input("Precio de Inicio (USD)", value=float(df_filtrado["Precio_Inicio_Mes"].iloc[-1]))
        volatilidad = st.slider("Volatilidad Esperada (%)", 0.0, 50.0, float(df_filtrado["Volatilidad_Media_Mensual"].mean()))
        volumen = st.number_input("Volumen Promedio", value=float(df_filtrado["Volumen_Promedio_Mensual"].mean()))

        input_data = np.array([[p_inicio, volatilidad, volumen]])
        prediccion_final = model.predict(input_data)[0]

        st.metric("Precio Estimado Fin de Mes", f"${prediccion_final:,.2f}", f"Err. Promedio: ±${mae:,.2f}")

    with col2:
        st.subheader("Visualización de la Regresión Lineal")
        # Gráfico de Dispersión + Línea de Regresión
        fig = go.Figure()

        # Datos Reales
        fig.add_trace(go.Scatter(x=df_filtrado["Precio_Inicio_Mes"], y=y, 
                                 mode='markers', name='Datos Históricos',
                                 marker=dict(color='#1f77b4', size=10)))

        # Línea de Tendencia
        fig.add_trace(go.Scatter(x=df_filtrado["Precio_Inicio_Mes"], y=y_pred, 
                                 mode='lines', name='Línea de Regresión',
                                 line=dict(color='red', width=3)))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Precio Inicio de Mes (USD)",
            yaxis_title="Precio Fin de Mes (USD)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tabla de Datos
    with st.expander("Ver Datos de Entrenamiento (Capa Gold)"):
        st.write(df_filtrado)

except Exception as e:
    st.error(f"Error al cargar datos o entrenar el modelo: {e}")
    st.info("Asegurate de que el archivo 'gold_data.csv' esté en el repositorio.")

st.markdown("---")
st.caption(f"Nicolas - {pd.Timestamp.now().strftime('%d/%m/%Y')} | Arquitectura Medallion & Machine Learning")