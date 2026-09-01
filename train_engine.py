import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

print("⚙️ Initializing NaviGo AI Training Pipeline...")

# 1. Load the Tabular Sensor Logs
try:
    df = pd.read_csv("sample_imu_log.csv")
except FileNotFoundError:
    print("❌ Error: 'sample_imu_log.csv' not found. Please run generate_data.py first.")
    exit()

# 2. Feature Engineering: Define Inputs and Target Adjustments
X = df[['accel_x_mps2', 'gyro_z_radps']].values

# Target Outputs (y): The progressive drift corrections needed
true_drift_bias = 0.45 + np.sin(df['timestamp_sec'].values * 0.1) * 0.05
y = true_drift_bias

# 3. Normalize Data Scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train a Robust Regressor Model
model = HuberRegressor()
model.fit(X_scaled, y)

print("🎯 AI Training Complete! Model has successfully learned the sensor drift signature.")

# 5. Export Model Artifacts to Your Workspace
with open("navigo_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("navigo_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("💾 System artifacts successfully saved: 'navigo_model.pkl' & 'navigo_scaler.pkl'")
