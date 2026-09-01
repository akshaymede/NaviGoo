import pandas as pd
import numpy as np

total_seconds = 20  # Longer simulation run
frequency = 10  
total_rows = total_seconds * frequency

timestamps = np.linspace(0, total_seconds, total_rows)

# --- ADDING SPACE REGOLITH TERRAIN VIBRATION (HIGH NOISE BURSTS) ---
terrain_vibration_noise = np.random.normal(0, 0.4, total_rows) 
accel_x = np.ones(total_rows) * 0.6 + terrain_vibration_noise
gyro_z = np.sin(timestamps * 0.4) * 0.15 + np.random.normal(0, 0.03, total_rows)

true_lat = 17.4481 + (timestamps * 0.00015)
true_lon = 78.3741 + (np.sin(timestamps * 0.25) * 0.00012)

imu_dataset = pd.DataFrame({
    'timestamp_sec': timestamps,
    'accel_x_mps2': accel_x,
    'gyro_z_radps': gyro_z,
    'gps_latitude': true_lat,
    'gps_longitude': true_lon
})

imu_dataset.to_csv("sample_imu_log.csv", index=False)
print("🎯 Advanced terrain simulation dataset written to 'sample_imu_log.csv' successfully!")
