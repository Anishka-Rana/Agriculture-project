from flask import render_template
import random

def setup_weather_routes(app):
    # Keep your mock data, but render it from app.py's /weather_forecasting
    locations = [
        {"name": "Chandigarh", "temp": f"{random.randint(20, 35)}°C", "desc": "Partly Cloudy"},
        {"name": "Ludhiana", "temp": f"{random.randint(18, 33)}°C", "desc": "Sunny"},
        {"name": "Pathankot", "temp": f"{random.randint(15, 30)}°C", "desc": "Light Rain"},
    ]

    @app.route("/weather/predict", methods=["GET"])
    def weather_predict():
        return {"locations": locations}
