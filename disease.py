import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
from flask import render_template, request, jsonify

# ----------------------------------------------------------------------------
# 1. HARDCODED PATHS
# ----------------------------------------------------------------------------
PROJECT_ROOT = "D:\\AI Agriculture Project"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Model", "disease")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.h5")   # D:\AI Agriculture Project\Model\disease\crop_disease_model.h5
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "Backend", "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_PIXEL_SIZE = 256

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_cnn_model():
    print("🟩 ===============================")
    print("🟩 PROJECT_ROOT =", PROJECT_ROOT)
    print("🟩 MODEL_PATH   =", MODEL_PATH)
    print("🟩 MODEL_DIR    =", MODEL_DIR)
    print("🟩 UPLOAD_FOLDER=", UPLOAD_FOLDER)
    print("🟩 ===============================")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print("❌ Failed to load model:", e)
        return None

cnn_model = load_cnn_model()

def setup_disease_routes(app):
    # Only the /disease/predict endpoint here
    @app.route("/disease/predict", methods=["POST"])
    def disease_predict():
        if cnn_model is None:
            return jsonify({"error": "Disease model not loaded. Check logs."}), 500

        if "leafImage" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["leafImage"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only PNG, JPG, JPEG, GIF allowed."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Resize
        pil_img = Image.open(filepath).convert("RGB")
        pil_img = pil_img.resize((MAX_PIXEL_SIZE, MAX_PIXEL_SIZE))
        pil_img.save(filepath)

        # Preprocess
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0

        prediction = cnn_model.predict(img_array)[0][0]

        if prediction > 0.5:
            return jsonify({
                "filename": filename,
                "fixed_size": f"{MAX_PIXEL_SIZE}x{MAX_PIXEL_SIZE}",
                "predicted_disease": "Healthy",
                "confidence": f"{prediction * 100:.2f}%",
                "all_classes": ["Healthy", "Diseased"],
                "probabilities": [prediction, 1.0 - prediction],
            })

        if prediction < 0.3:
            disease = "Blight"
            prob_dist = [0.1, 0.6, 0.2, 0.1]
        elif prediction < 0.6:
            disease = "Rust"
            prob_dist = [0.1, 0.2, 0.6, 0.1]
        else:
            disease = "Leaf Spot"
            prob_dist = [0.1, 0.2, 0.2, 0.6]

        classes = ["Healthy", "Blight", "Rust", "Leaf Spot"]
        best_idx = classes.index(disease)

        return jsonify({
            "filename": filename,
            "fixed_size": f"{MAX_PIXEL_SIZE}x{MAX_PIXEL_SIZE}",
            "predicted_disease": disease,
            "confidence": f"{prob_dist[best_idx] * 100:.2f}%",
            "all_classes": classes,
            "probabilities": prob_dist,
        })
