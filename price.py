import os
import joblib
from flask import render_template, request, jsonify

# ----------------------------------------------------------------------------
# 1. HARDCODED PATHS (MATCH YOUR ACTUAL FILES)
# ----------------------------------------------------------------------------
PROJECT_ROOT = "D:\\AI Agriculture Project"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Model", "price")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_price_model.pkl")
MONTH_ENCODER_PATH = os.path.join(MODEL_DIR, "month_encoder.pkl")
LOCATION_ENCODER_PATH = os.path.join(MODEL_DIR, "location_encoder.pkl")   # ✅ fixed
CROP_ENCODER_PATH = os.path.join(MODEL_DIR, "crop_encoder.pkl")
ALLOWED_CROPS = [
    "Wheat", "Rice", "Maize", "Paddy", "Potato", "Tomato", "Onion", "Chilli",
    "Brinjal", "Cabbage", "Cauliflower", "Peas", "Beans", "Cucumber", "Spinach",
    "Carrot", "Garlic", "Mustard", "Peanut", "Soybean", "Banana", "Apple",
    "Orange", "Mango", "Grapes", "Coconut", "Sugarcane", "Cotton", "Tea", "Coffee"
]

def load_price_model_and_encoders():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None

    if not os.path.exists(MONTH_ENCODER_PATH) or not os.path.exists(LOCATION_ENCODER_PATH) or not os.path.exists(CROP_ENCODER_PATH):
        return None, None, None, None

    try:
        model = joblib.load(MODEL_PATH)
        month_encoder = joblib.load(MONTH_ENCODER_PATH)
        location_encoder = joblib.load(LOCATION_ENCODER_PATH)
        crop_encoder = joblib.load(CROP_ENCODER_PATH)

        print("✅ Price model and encoders loaded.")
        return model, month_encoder, location_encoder, crop_encoder
    except Exception as e:
        print("❌ Failed to load price model or encoders:", e)
        import traceback
        traceback.print_exc()
        return None, None, None, None

price_model, month_encoder, location_encoder, crop_encoder = load_price_model_and_encoders()

def preprocess_input(data):
    import pandas as pd

    month = data.get("month", "January")
    location = data.get("location", "Punjab")
    raw_crop = data.get("crop_type", "Wheat")

    # Use uppercase crop as in training
    crop_type = raw_crop.title()  # "wheat" → "Wheat", "WHEAT" → "Wheat"

    if crop_type not in ALLOWED_CROPS:
        print("❌ Invalid crop type:", crop_type)
        return None

    df = pd.DataFrame([{
        "Month": month,
        "Location": location,
        "Crop": crop_type,
    }])

    try:
        df["Month_encoded"] = month_encoder.transform(df["Month"])
        df["Location_encoded"] = location_encoder.transform(df["Location"])
        df["Crop_encoded"] = crop_encoder.transform(df["Crop"])
    except Exception as e:
        print("❌ Encoder error:", e)
        return None

    return df[["Month_encoded", "Location_encoded", "Crop_encoded"]]

def make_price_prediction(inputs):
    if not all(k in inputs for k in ["month", "location", "crop_type"]):
        return {"error": "Missing required fields: month, location, crop_type."}

    df = preprocess_input(inputs)
    if df is None:
        return {"error": "Invalid crop type or data."}

    try:
        price = price_model.predict(df)[0]
        return {"predicted_price": round(float(price), 2)}
    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}

def setup_price_routes(app):
    @app.route("/price/predict", methods=["POST"])
    def price_predict():
        if not all(x is not None for x in [price_model, month_encoder, location_encoder, crop_encoder]):
            return jsonify({"error": "Price model or encoders not loaded. Check logs."}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        result = make_price_prediction(data)
        status = 400 if "error" in result else 200
        return jsonify(result), status
