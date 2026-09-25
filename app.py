from flask import Flask, render_template, request
import joblib
import pandas as pd

from feature_extraction import extract_features


app = Flask(__name__)

# Load model Random Forest
model = joblib.load("model/random_forest.pkl")


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
                probabilities[prediction] * 100, 2
            )

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


if __name__ == "__main__":
    app.run(debug=True)