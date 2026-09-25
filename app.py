from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

from feature_extraction import extract_features


app = Flask(__name__)


# ==============================
# LOAD MODEL RANDOM FOREST
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "random_forest.pkl"
)

model = joblib.load(MODEL_PATH)


# ==============================
# ROUTE UTAMA
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    url_input = ""

    if request.method == "POST":

        url_input = request.form.get("url", "").strip()

        if url_input:

            # Ekstraksi fitur URL
            features = extract_features(url_input)

            # Ubah fitur menjadi DataFrame
            features_df = pd.DataFrame([features])

            # Prediksi kelas
            prediction = model.predict(features_df)[0]

            # Probabilitas prediksi
            probabilities = model.predict_proba(features_df)[0]

            # Ambil probabilitas dari kelas yang diprediksi
            confidence = round(
                probabilities[prediction] * 100,
                2
            )

            # Hasil prediksi
            if prediction == 1:
                result = "TERINDIKASI JUDI ONLINE"
            else:
                result = "TIDAK TERINDIKASI JUDI ONLINE"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        url=url_input
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(debug=True)