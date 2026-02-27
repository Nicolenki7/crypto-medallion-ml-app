# 🚀 Crypto Analytics & ML Predictor

**Medallion Architecture | Machine Learning | Interactive Dashboard**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This application demonstrates a complete **Medallion Architecture** implementation for cryptocurrency data processing, featuring:

- **Gold Layer Data**: Cleaned, aggregated, and business-ready datasets
- **Machine Learning**: Linear Regression model for price prediction
- **Interactive Dashboard**: Real-time predictions with Streamlit + Plotly
- **Production-Ready**: Error handling, caching, and responsive UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MEDALLION ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BRONZE LAYER (Raw Data)                                    │
│  └─→ Raw crypto prices from APIs (Binance, CoinGecko)       │
│                                                              │
│  SILVER LAYER (Cleaned)                                     │
│  └─→ Validated, deduplicated, type-corrected data           │
│                                                              │
│  GOLD LAYER (Business-Ready)                                │
│  └─→ Aggregated features for ML (monthly metrics)           │
│       - Precio_Inicio_Mes                                   │
│       - Precio_Fin_Mes                                      │
│       - Volatilidad_Media_Mensual                           │
│       - Volumen_Promedio_Mensual                            │
│                                                              │
│  ML LAYER (Predictions)                                     │
│  └─→ Linear Regression → Precio_Fin_Mes prediction          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Features

### Machine Learning Model
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: 
  - Opening price (monthly)
  - Average volatility
  - Average volume
- **Target**: Month-end closing price
- **Metrics**: R² Score, MAE (Mean Absolute Error)

### Interactive Dashboard
- **Asset Selection**: Choose from multiple cryptocurrencies
- **Input Simulation**: Adjust parameters for "what-if" analysis
- **Visualizations**: 
  - Regression scatter plot
  - Historical data table
  - Performance metrics

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Nicolenki7/crypto-medallion-ml-app.git
cd crypto-medallion-ml-app

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
crypto-medallion-ml-app/
├── app.py                      # Main Streamlit application
├── gold_data.csv               # Gold layer dataset (aggregated features)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📊 Data Pipeline

### Gold Layer Features

| Feature | Description | Type |
|---------|-------------|------|
| `Nombre_Moneda` | Cryptocurrency name | String |
| `Precio_Inicio_Mes` | Month opening price (USD) | Float |
| `Precio_Fin_Mes` | Month closing price (USD) | Float |
| `Volatilidad_Media_Mensual` | Average monthly volatility (%) | Float |
| `Volumen_Promedio_Mensual` | Average monthly volume | Float |

### Model Performance

- **R² Score**: ~0.98 (Bitcoin)
- **Mean Absolute Error**: Varies by asset
- **Features Importance**: Price start > Volume > Volatility

---

## 🎨 Usage Guide

### 1. Select Cryptocurrency
Use the sidebar dropdown to choose your target asset (BTC, ETH, etc.)

### 2. Configure Prediction Inputs
- **Precio de Inicio**: Current/expected month opening price
- **Volatilidad Esperada**: Expected volatility percentage (0-50%)
- **Volumen Promedio**: Expected average trading volume

### 3. View Results
- **Estimated Price**: Model prediction for month-end
- **Error Margin**: Average prediction error (MAE)
- **Regression Plot**: Visual representation of model fit

---

## 🔮 Future Enhancements

- [ ] Multi-model comparison (XGBoost, Random Forest, LSTM)
- [ ] Real-time data ingestion from crypto APIs
- [ ] Feature engineering (technical indicators, sentiment)
- [ ] Model retraining pipeline (MLflow integration)
- [ ] Backtesting framework
- [ ] Alert system for price predictions

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | Plotly |
| **Deployment** | Streamlit Cloud |

---

## 📝 Data Engineering Best Practices

### 1. Medallion Architecture
- Clear separation between raw, cleaned, and business-ready data
- Each layer has specific quality standards and transformations

### 2. Feature Engineering
- Domain-specific features (volatility, volume patterns)
- Aggregation at appropriate time granularity (monthly)

### 3. Model Validation
- R² score for goodness of fit
- MAE for practical error interpretation
- Train/test split for generalization assessment

### 4. Code Quality
- Type hints and docstrings
- Error handling for production robustness
- Caching for performance optimization

---

## 👨‍💻 Author

**Nicolas Zalazar**  
*Senior Data Engineer | Microsoft Fabric & Snowflake Specialist*

- 📧 zalazarn046@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/nicolas-zalazar-63340923a)
- 🐙 [GitHub](https://github.com/Nicolenki7)
- 📊 [Kaggle](https://www.kaggle.com/nicolaszalazar73)

### Core Competencies
- **Data Engineering**: ETL/ELT, Medallion Architecture, Data Modeling
- **Cloud Platforms**: Microsoft Fabric, Snowflake, Databricks, AWS
- **Machine Learning**: Predictive Modeling, Feature Engineering, scikit-learn
- **Programming**: Python (Pandas, PySpark), SQL (Advanced)
- **BI & Visualization**: Power BI, Tableau, Streamlit, Plotly

---

## 📄 License

MIT License — Feel free to fork, modify, and use for personal or commercial projects.

---

*Last Updated: February 2026*
