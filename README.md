# 🚀 Crypto Medallion ML App

**Medallion Architecture for DeFi | Machine Learning Price Prediction | Interactive Streamlit Dashboard**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

Complete **Medallion Architecture** implementation for cryptocurrency analytics, featuring gold-layer data processing, machine learning price prediction, and an interactive Streamlit dashboard for real-time analysis.

This project demonstrates production-ready data engineering patterns for DeFi analytics with clear separation between raw, cleaned, and business-ready data layers.

---

## 💼 Business Impact

- **Signal-vs-Noise Analysis**: Filters market volatility to identify actionable price trends
- **Volume-Backed Validation**: Correlates price predictions with trading volume patterns
- **What-If Scenario Planning**: Interactive parameter adjustment for investment decision support
- **Production-Ready ML**: R² ~0.98 accuracy on Bitcoin price predictions

---

## 🛠️ Technical Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (Linear Regression) |
| **Visualization** | Plotly |
| **Data Architecture** | Medallion (Bronze/Silver/Gold) |
| **Deployment** | Streamlit Cloud |

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

## 🚀 Key Features

### Machine Learning Model
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: Opening price, average volatility, average volume
- **Target**: Month-end closing price
- **Performance**: R² ~0.98 (Bitcoin), MAE tracked per asset

### Interactive Dashboard
- **Multi-Asset Support**: BTC, ETH, and major cryptocurrencies
- **Parameter Simulation**: Adjust inputs for scenario analysis
- **Visual Analytics**: Regression plots, historical tables, metrics
- **Production Features**: Error handling, caching, responsive UI

---

## 📊 Results & Metrics

| Metric | Value |
| :--- | :--- |
| **R² Score (BTC)** | ~0.98 |
| **Feature Importance** | Price Start > Volume > Volatility |
| **Data Granularity** | Monthly aggregated metrics |
| **Assets Supported** | Multiple cryptocurrencies |

---

## 📁 Project Structure

```
crypto-medallion-ml-app/
├── app.py                      # Main Streamlit application
├── gold_data.csv               # Gold layer dataset (aggregated features)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔧 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Nicolenki7/crypto-medallion-ml-app.git
cd crypto-medallion-ml-app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📈 Usage

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

## 🎯 Key Learnings

- **Medallion Architecture** provides clear data quality boundaries
- **Feature engineering** (volatility, volume) improves prediction accuracy
- **Interactive dashboards** enable business users to explore ML outputs
- **Gold-layer aggregation** reduces noise for trend analysis

---

## 🔮 Future Enhancements

- [ ] Multi-model comparison (XGBoost, Random Forest, LSTM)
- [ ] Real-time data ingestion from crypto APIs
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] MLflow integration for model tracking
- [ ] Backtesting framework
- [ ] Alert system for price predictions

---

## 🔗 Links

| Resource | URL |
| :--- | :--- |
| **Repository** | https://github.com/Nicolenki7/crypto-medallion-ml-app |
| **Live Demo** | (Deploy on Streamlit Cloud) |
| **Related Project** | [Crypto Medallion Analytics](https://github.com/Nicolenki7/Crypto_Medallion_Analytics) |

---

## 📝 License

MIT License — Feel free to fork, modify, and use for personal or commercial projects.

---

## 👤 Author

**Nicolás Zalazar** | Senior Data Engineer & Microsoft Fabric Specialist

- GitHub: [@Nicolenki7](https://github.com/Nicolenki7)
- LinkedIn: [nicolas-zalazar-63340923a](https://www.linkedin.com/in/nicolas-zalazar-63340923a)
- Portfolio: [nicolenki7.github.io/Portfolio](https://nicolenki7.github.io/Portfolio/)
- Email: zalazarn046@gmail.com

---

*Last Updated: March 2026*
