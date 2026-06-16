# Model/price/train_price_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Paths (from Model/price/)
DATA_PATH = "crop_data.csv"
MODEL_DIR = "."

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset head:")
print(df.head())

# Check columns
assert "Month" in df.columns, "Missing 'Month' column"
assert "Location" in df.columns, "Missing 'Location' column"
assert "Crop" in df.columns, "Missing 'Crop' column"
assert "Price" in df.columns, "Missing 'Price' column"

# Encode categorical features
month_encoder = LabelEncoder()
location_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df["Month_encoded"] = month_encoder.fit_transform(df["Month"])
df["Location_encoded"] = location_encoder.fit_transform(df["Location"])
df["Crop_encoded"] = crop_encoder.fit_transform(df["Crop"])

# Features and target
X = df[["Month_encoded", "Location_encoded", "Crop_encoded"]]
y = df["Price"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("MAE:", mae)
print("RMSE:", rmse)

# Save model and encoders
model_path = os.path.join(MODEL_DIR, "crop_price_model.pkl")
month_path = os.path.join(MODEL_DIR, "month_encoder.pkl")
location_path = os.path.join(MODEL_DIR, "location_encoder.pkl")
crop_path = os.path.join(MODEL_DIR, "crop_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(month_encoder, month_path)
joblib.dump(location_encoder, location_path)
joblib.dump(crop_encoder, crop_path)

print(f"Model and encoders saved to {MODEL_DIR}")

