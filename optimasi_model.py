import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
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

print("=" * 60)
print("OPTIMASI RANDOM FOREST")
print("=" * 60)

data = pd.read_csv("dataset/dataset_url.csv")

print(f"\nTotal dataset: {len(data)} URL")


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
# 3. SPLIT DATA 80:20
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPembagian dataset:")
print(f"Training : {len(X_train)}")
print(f"Testing  : {len(X_test)}")


# ==========================================
# 4. PARAMETER YANG AKAN DIUJI
# ==========================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# ==========================================
# 5. MODEL RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    random_state=42
)


# ==========================================
# 6. GRID SEARCH
# ==========================================

print("\nMemulai optimasi parameter...")
print("Mohon tunggu...")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


# ==========================================
# 7. PARAMETER TERBAIK
# ==========================================

print("\n")
print("=" * 60)
print("PARAMETER TERBAIK")
print("=" * 60)

print(grid_search.best_params_)

print(f"\nBest CV Accuracy: "
      f"{grid_search.best_score_ * 100:.2f}%")


# ==========================================
# 8. EVALUASI PADA DATA TESTING
# ==========================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("\n")
print("=" * 60)
print("HASIL EVALUASI DATA TESTING")
print("=" * 60)

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
# 9. SIMPAN MODEL TERBAIK
# ==========================================

joblib.dump(
    best_model,
    "model/random_forest.pkl"
)

print("\n")
print("=" * 60)
print("MODEL TERBAIK BERHASIL DISIMPAN")
print("=" * 60)

print("Lokasi:")
print("model/random_forest.pkl")