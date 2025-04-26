# Time-Series-Challenge
This project aims to accurately forecast transformer oil temperature (OT) for the next 24 hours at 1-hour intervals using a GRU-based deep learning approach. Forecasting OT is crucial in preventing equipment failures, managing electrical loads, and ensuring energy distribution efficiency.

Project Highlights
📊 Multivariate Time Series Forecasting using GRU (Gated Recurrent Units)

🔁 Predicts next 24 OT values in one shot (multi-output forecasting)

⚙️ Includes feature engineering (lags, rolling stats, time features)

📉 Evaluates model using MAE, RMSE, MAPE, SMAPE

🔍 Includes EDA to analyze load patterns and extreme transformer behavior

📁 Outputs 24-hour forecasts in CSV and visual plots

Model Overview
Architecture: Stacked GRU with Dropout

Input: 24 time steps of multivariate features

Output: Next 24 hourly oil temperature values

Optimizer: Adam with learning rate 0.0005

Loss Function: Mean Squared Error (MSE)

Early Stopping: Prevents overfitting based on validation loss
