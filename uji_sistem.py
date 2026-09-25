import pandas as pd
import joblib

from feature_extraction import extract_features


# Load model
model = joblib.load("model/random_forest.pkl")

# Load dataset
data = pd.read_csv("dataset/dataset_url.csv")


print("=" * 70)
print("PENGUJIAN SISTEM DETEKSI LINK JUDI ONLINE")
print("=" * 70)

benar = 0
total = len(data)

hasil_pengujian = []


for index, row in data.iterrows():

    url = row["url"]
    label_asli = int(row["label"])

    # Ekstraksi fitur
    features = extract_features(url)

    # Ubah fitur menjadi dataframe
    X = pd.DataFrame([features])

    # Prediksi
    prediksi = int(model.predict(X)[0])

    # Confidence
    probabilitas = model.predict_proba(X)[0]
    confidence = max(probabilitas) * 100

    # Cek benar / salah
    if prediksi == label_asli:
        status = "BENAR"
        benar += 1
    else:
        status = "SALAH"

    # Ubah label menjadi teks
    if prediksi == 1:
        hasil_prediksi = "TERINDIKASI JUDI ONLINE"
    else:
        hasil_prediksi = "TIDAK TERINDIKASI JUDI ONLINE"

    if label_asli == 1:
        hasil_asli = "JUDI ONLINE"
    else:
        hasil_asli = "BUKAN JUDI"

    hasil_pengujian.append({
        "url": url,
        "label_asli": hasil_asli,
        "prediksi": hasil_prediksi,
        "confidence": round(confidence, 2),
        "status": status
    })


# Buat DataFrame hasil
hasil_df = pd.DataFrame(hasil_pengujian)


# Tampilkan hasil
print("\nHASIL PENGUJIAN:")
print("=" * 70)

for i, row in hasil_df.iterrows():

    print(f"\nPengujian {i + 1}")
    print("-" * 40)
    print(f"URL        : {row['url']}")
    print(f"Label Asli : {row['label_asli']}")
    print(f"Prediksi   : {row['prediksi']}")
    print(f"Confidence : {row['confidence']}%")
    print(f"Status     : {row['status']}")


# Hitung akurasi
akurasi = (benar / total) * 100


print("\n")
print("=" * 70)
print("HASIL AKHIR")
print("=" * 70)

print(f"Total data       : {total}")
print(f"Prediksi benar   : {benar}")
print(f"Prediksi salah   : {total - benar}")
print(f"Akurasi pengujian: {akurasi:.2f}%")

print("=" * 70)


# Simpan hasil ke CSV
hasil_df.to_csv("hasil_pengujian.csv", index=False)

print("\nHasil pengujian disimpan ke: hasil_pengujian.csv")