from flask import Flask, render_template
from routes import price, disease, weather

app = Flask(__name__)

# Register routes
price.setup_price_routes(app)
disease.setup_disease_routes(app)
weather.setup_weather_routes(app)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/price_prediction")
def price_prediction():
    return render_template("price_prediction.html")

@app.route("/disease_detection")
def disease_detection():
    return render_template("disease_detection.html")

@app.route("/weather_forecasting")
def weather_forecasting():
    return render_template("weather_forecasting.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)

