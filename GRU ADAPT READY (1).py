#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[2]:


# Load and preprocess the data
df = pd.read_csv("C:\\Users\\Shreya\\Downloads\\train.csv", parse_dates=['date'])
df.fillna(method='ffill', inplace=True)


# In[3]:


# Feature engineering
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['OT_lag1'] = df['OT'].shift(1)
df['OT_lag2'] = df['OT'].shift(2)
df['OT_roll_mean_3'] = df['OT'].rolling(window=3).mean()
df['OT_roll_mean_6'] = df['OT'].rolling(window=6).mean()
df.fillna(method='bfill', inplace=True)


# In[4]:


# Define features and target
feature_columns = [
    'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',
    'hour', 'dayofweek', 'OT_lag1', 'OT_lag2', 'OT_roll_mean_3', 'OT_roll_mean_6'
]
target_column = 'OT'
forecast_horizon = 24
time_steps = 24


# In[5]:


# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[feature_columns + [target_column]])
X = scaled_data[:, :-1]
y = scaled_data[:, -1]


# In[6]:


# Create time series dataset
def create_time_series_dataset(X, y, time_steps, forecast_horizon):
    X_new, y_new = [], []
    for i in range(len(X) - time_steps - forecast_horizon + 1):
        X_new.append(X[i:i + time_steps])
        y_new.append(y[i + time_steps:i + time_steps + forecast_horizon])
    return np.array(X_new), np.array(y_new)

X_new, y_new = create_time_series_dataset(X, y, time_steps, forecast_horizon)


# In[7]:


# Split into training and validation sets
split_idx = int(len(X_new) * 0.8)
X_train, X_val = X_new[:split_idx], X_new[split_idx:]
y_train, y_val = y_new[:split_idx], y_new[split_idx:]


# In[8]:


# Build improved GRU model
model = Sequential()
model.add(GRU(64, activation='relu', return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(32, activation='relu'))
model.add(Dense(forecast_horizon))


# In[9]:


model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# In[10]:


# Train the model
history = model.fit(X_train, y_train, epochs=150, batch_size=16,
                    validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)


# In[11]:


# Plot training/validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()


# In[12]:


# === Evaluation Metrics ===
def inverse_ot_scaling(y_scaled):
    ot_min = scaler.data_min_[-1]
    ot_max = scaler.data_max_[-1]
    return y_scaled * (ot_max - ot_min) + ot_min

def safe_mape(y_true, y_pred, epsilon=1e-5):
    mask = y_true > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1e-5, denominator)
    return np.mean(diff) * 100


# In[13]:


# Predict on validation
y_pred = model.predict(X_val)
y_val_actual = inverse_ot_scaling(y_val)
y_pred_actual = inverse_ot_scaling(y_pred)


# In[14]:


# Evaluation
mae = mean_absolute_error(y_val_actual.flatten(), y_pred_actual.flatten())
rmse = np.sqrt(mean_squared_error(y_val_actual.flatten(), y_pred_actual.flatten()))
mape = safe_mape(y_val_actual.flatten(), y_pred_actual.flatten())
smape_score = smape(y_val_actual.flatten(), y_pred_actual.flatten())

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"sMAPE: {smape_score:.2f}%")


# In[16]:


X_last = X[-time_steps:]
X_last = np.expand_dims(X_last, axis=0)
forecast_scaled = model.predict(X_last)
forecast_actual = inverse_ot_scaling(forecast_scaled)[0]

# Forecast timestamps
last_timestamp = df['date'].iloc[-1]
forecast_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1),
                                    periods=forecast_horizon, freq='H')

# Forecast DataFrame
forecast_df = pd.DataFrame({
    'timestamp': forecast_timestamps,
    'predicted_OT': forecast_actual
})

# === Save and Display Forecast Safely ===
import os
os.makedirs("outputs", exist_ok=True)  # Ensure the outputs folder exists

# Save the forecast
forecast_df.to_csv('outputs/improved_24_hour_forecast.csv', index=False)
print("\n✅ 24-hour forecast saved to 'outputs/improved_24_hour_forecast.csv'")
print("\n📅 24-Hour Forecast:")
print(forecast_df)


# In[17]:


# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(forecast_df['timestamp'], forecast_df['predicted_OT'], marker='o', color='tab:blue')
plt.title("Improved 24-Hour OT Forecast")
plt.xlabel("Timestamp")
plt.ylabel("Predicted OT")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[19]:


#Exploratory Data Analysis (EDA)
# To determine which load types impact OT the most (likely HUFL, MUFL).
import seaborn as sns

load_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
corr = df[load_columns + ['OT']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Loads and Oil Temperature")
plt.show()


# In[21]:


#to check if the temperature follows a normal range or has extreme outliers.
plt.figure(figsize=(6, 4))
sns.histplot(df['OT'], bins=50, kde=True, color='steelblue')
plt.title("Distribution of OT")
plt.xlabel("OT (°C)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[22]:


#Extreme Load Capacity Evaluation
#Compare OT in High vs Low Load Periods
# Define "high load" as when HUFL > 95th percentile
high_load_threshold = df['HUFL'].quantile(0.95)
high_load_df = df[df['HUFL'] > high_load_threshold]
low_load_df = df[df['HUFL'] <= high_load_threshold]

print("High Load OT Mean:", high_load_df['OT'].mean())
print("Low Load OT Mean:", low_load_df['OT'].mean())


# In[ ]:




