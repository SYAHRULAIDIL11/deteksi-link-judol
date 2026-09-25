import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from feature_extraction import extract_features


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Membaca dataset...")

data = pd.read_csv("dataset/dataset_url.csv")

print(f"Total dataset: {len(data)} URL")


# ==========================================
# 2. EKSTRAKSI FITUR
# ==========================================

print("\nMelakukan ekstraksi fitur...")

features = []

for url in data["url"]:
    features.append(extract_features(url))

X = pd.DataFrame(features)
y = data["label"]


# ==========================================
# 3. PEMBAGIAN DATA 80:20
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPembagian dataset:")
print(f"Data training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")


# ==========================================
# 4. MEMBUAT MODEL RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None
)

model.fit(X_train, y_train)

print("Training selesai!")


# ==========================================
# 5. PREDIKSI DATA TESTING
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. EVALUASI MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 50)
print("HASIL EVALUASI MODEL")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Tidak Terindikasi Judi",
            "Terindikasi Judi"
        ]
    )
)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 7. SIMPAN MODEL
# ==========================================

joblib.dump(model, "model/random_forest.pkl")

print("\nModel berhasil disimpan:")
print("model/random_forest.pkl")