# 🔋 Transformer Oil Temperature Forecasting (24-Hour Horizon)

This project aims to accurately forecast transformer **oil temperature (OT)** for the next 24 hours at **1-hour intervals** using a GRU-based deep learning approach. Forecasting OT is crucial in preventing equipment failures, managing electrical loads, and ensuring energy distribution efficiency.

---

## 📌 Project Highlights

- 📊 **Multivariate Time Series Forecasting** using GRU (Gated Recurrent Units)  
- ♻️ Predicts next 24 OT values in one shot (multi-output forecasting)  
- ⚙️ Includes **feature engineering** (lags, rolling stats, time features)  
- 📉 Evaluates model using **MAE, RMSE, MAPE, SMAPE**  
- 🔍 Includes **EDA** to analyze load patterns and extreme transformer behavior  
- 📁 Outputs 24-hour forecasts in CSV and visual plots

---

## 🧠 Model Overview

- **Architecture**: Stacked GRU with Dropout  
- **Input**: 24 time steps of multivariate features  
- **Output**: Next 24 hourly oil temperature values  
- **Optimizer**: Adam with learning rate 0.0005  
- **Loss Function**: Mean Squared Error (MSE)  
- **Early Stopping**: Prevents overfitting based on validation loss

---

## 📂 Dataset

- File: `train.csv`  
- Features include:
  - Load values: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`
  - Time info: `hour`, `dayofweek`
  - Engineered: `OT_lag1`, `OT_lag2`, `OT_roll_mean_3`, `OT_roll_mean_6`

---

## 📈 Evaluation Metrics

On validation set:

- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **MAPE** (Mean Absolute Percentage Error)  
- **SMAPE** (Symmetric Mean Absolute Percentage Error)

---

## 📊 Forecast Output

- File: `outputs/improved_24_hour_forecast.csv`  
- Columns: `timestamp`, `predicted_OT`  
- Also visualized using a line plot

---

## 📌 Exploratory Data Analysis (EDA)

- **Correlation Heatmap** to explore impact of load types on OT  
- **Distribution Plot** for understanding OT value ranges  
- **Extreme Load Analysis** using HUFL percentiles to compare OT under stress

---

## 🚀 Getting Started

1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```
3. Run the script:  
   ```bash
   python GRU_ADAPT_READY.py
   ```

---

## 📁 Outputs

- ✅ `improved_24_hour_forecast.csv`  
- 📈 Forecast plot  
- 📊 Evaluation metrics and EDA visualizations

---





