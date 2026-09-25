import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# BACA HASIL PENGUJIAN
# ==============================

file = "hasil_pengujian.csv"

df = pd.read_csv(file)

# Rapikan nama kolom
df.columns = df.columns.str.strip().str.lower()

print("Kolom dataset:")
print(df.columns.tolist())

# ==============================
# HITUNG HASIL PENGUJIAN
# ==============================

total = len(df)

benar = (df["status"].astype(str).str.upper() == "BENAR").sum()
salah = (df["status"].astype(str).str.upper() == "SALAH").sum()

akurasi = (benar / total) * 100

print("\n======================================")
print("HASIL PENGUJIAN SISTEM")
print("======================================")
print(f"Total data       : {total}")
print(f"Prediksi benar   : {benar}")
print(f"Prediksi salah   : {salah}")
print(f"Akurasi sistem   : {akurasi:.2f}%")
print("======================================")


# ==============================
# GRAFIK PREDIKSI BENAR & SALAH
# ==============================

plt.figure(figsize=(7, 5))

kategori = ["Prediksi Benar", "Prediksi Salah"]
jumlah = [benar, salah]

bars = plt.bar(kategori, jumlah)

plt.title("Hasil Pengujian Sistem Random Forest")
plt.ylabel("Jumlah URL")

for bar, nilai in zip(bars, jumlah):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3,
        str(nilai),
        ha="center",
        fontsize=13
    )

plt.tight_layout()

plt.savefig(
    "hasil_pengujian_sistem.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==============================
# GRAFIK AKURASI
# ==============================

plt.figure(figsize=(6, 5))

bars = plt.bar(
    ["Random Forest"],
    [akurasi / 100]
)

plt.ylim(0, 1)

plt.ylabel("Accuracy")
plt.title("Akurasi Pengujian Sistem")

plt.text(
    0,
    (akurasi / 100) + 0.02,
    f"{akurasi:.2f}%",
    ha="center",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "akurasi_pengujian_sistem.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()