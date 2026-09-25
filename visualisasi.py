import matplotlib.pyplot as plt
import numpy as np

# ==============================
# DATA HASIL EVALUASI
# ==============================

accuracy = 85.00

precision = {
    "Tidak Terindikasi Judi": 0.83,
    "Terindikasi Judi": 0.87
}

recall = {
    "Tidak Terindikasi Judi": 0.88,
    "Terindikasi Judi": 0.82
}

f1_score = {
    "Tidak Terindikasi Judi": 0.85,
    "Terindikasi Judi": 0.85
}

# Confusion Matrix
cm = np.array([
    [35, 5],
    [7, 33]
])

labels = [
    "Tidak Terindikasi Judi",
    "Terindikasi Judi"
]


# ==============================
# 1. CONFUSION MATRIX
# ==============================

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix Random Forest")
plt.colorbar()

plt.xticks([0, 1], labels, rotation=15)
plt.yticks([0, 1], labels)

plt.xlabel("Prediksi")
plt.ylabel("Label Aktual")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()


# ==============================
# 2. PRECISION, RECALL, F1
# ==============================

metrics = ["Precision", "Recall", "F1-Score"]

tidak_judi = [
    precision["Tidak Terindikasi Judi"],
    recall["Tidak Terindikasi Judi"],
    f1_score["Tidak Terindikasi Judi"]
]

judi = [
    precision["Terindikasi Judi"],
    recall["Terindikasi Judi"],
    f1_score["Terindikasi Judi"]
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    tidak_judi,
    width,
    label="Tidak Terindikasi Judi"
)

plt.bar(
    x + width / 2,
    judi,
    width,
    label="Terindikasi Judi"
)

plt.xticks(x, metrics)
plt.ylabel("Nilai")
plt.ylim(0, 1)

plt.title("Precision, Recall, dan F1-Score")

plt.legend()

plt.tight_layout()

plt.savefig("metrics_evaluation.png", dpi=300)

plt.show()


# ==============================
# 3. ACCURACY
# ==============================

plt.figure(figsize=(6, 5))

plt.bar(
    ["Random Forest"],
    [accuracy / 100]
)

plt.ylim(0, 1)

plt.ylabel("Accuracy")
plt.title("Accuracy Model Random Forest")

plt.text(
    0,
    accuracy / 100 + 0.02,
    f"{accuracy:.2f}%",
    ha="center",
    fontsize=14
)

plt.tight_layout()

plt.savefig("accuracy_model.png", dpi=300)

plt.show()

print("======================================")
print("VISUALISASI SELESAI")
print("======================================")
print("Accuracy       : 85.00%")
print("Confusion Matrix tersimpan")
print("Metrics tersimpan")
print("Accuracy tersimpan")