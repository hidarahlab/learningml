# ============================================================
# (OPSIONAL) MATIKAN WARNING MATMUL
# Biarkan ter-comment dulu. Run sekali tanpa ini.
# Kalau warning "encountered in matmul" MASIH muncul setelah
# pakai StandardScaler, baru hapus tanda # di 5 baris bawah.
# ============================================================

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning,
)


# ============================================================
# IMPORT LIBRARY
# ============================================================
import numpy as np, pandas as pd                                   # numpy: array & matematika, pandas: tabel (DataFrame)
from sklearn.model_selection import train_test_split               # alat buat bagi data train/test
from sklearn.linear_model import LogisticRegression                # algoritma model klasifikasi
from sklearn.metrics import accuracy_score, classification_report  # alat buat ngukur nilai model
from sklearn.preprocessing import StandardScaler                   # alat buat menyamakan skala fitur
from sklearn.pipeline import make_pipeline                         # alat buat merangkai beberapa langkah jadi satu


# ============================================================
# 1. BIKIN DATA SIMULASI
# ============================================================
rng = np.random.default_rng(42)   # generator angka acak, seed 42 biar hasilnya selalu sama tiap run
n = 10_000                        # jumlah transaksi (_ cuma pemisah biar gampang dibaca)

df = pd.DataFrame({                                       # bikin tabel dari dictionary {nama_kolom: isi}
    "nominal": rng.lognormal(12, 1, n).round(-3),         # nominal acak: banyak kecil, sesekali gede; dibulatkan ke ribuan
    "jam": rng.integers(0, 24, n),                        # jam transaksi 0-23 (24 nggak ikut)
    "trx_1jam_terakhir": rng.poisson(1, n),               # berapa kali transaksi dalam 1 jam terakhir, rata-rata 1
    "device_baru": rng.random(n) < 0.05,                  # True/False, ~5% transaksi pakai device baru
})

# skor kecurigaan (RAHASIA, model nggak dikasih tahu rumus ini)
skor = (
    (np.log(df.nominal) - 12)        # makin gede nominal, makin tinggi skor
    + (df.jam < 5) * 1.5             # transaksi jam 00-04 dapat +1.5
    + df.trx_1jam_terakhir * 0.8     # tiap transaksi dalam 1 jam terakhir +0.8
    + df.device_baru * 2.5           # pakai device baru +2.5
)

df["fraud"] = (skor + rng.normal(0, 1, n) > 5.5).astype(int)  # tambah noise acak, kalau skor > 5.5 = fraud (1), selain itu aman (0)

print(df.head())                              # tampilkan 5 baris pertama
print("Jumlah baris & kolom:", df.shape)      # (baris, kolom)
print("Rasio fraud:", df.fraud.mean())        # mean dari kolom 0/1 = proporsi fraud


# ============================================================
# 2. TRAINING
# ============================================================
X = df.drop(columns="fraud")   # FITUR: semua kolom kecuali fraud (yang dilihat model)
y = df.fraud                   # LABEL: kolom fraud saja (kunci jawaban)

X_train, X_test, y_train, y_test = train_test_split(   # bagi data jadi train & test
    X, y,
    test_size=0.2,       # 20% buat ujian (2.000), 80% buat belajar (8.000)
    stratify=y,          # jaga rasio fraud sama di train dan test (~1,5%)
    random_state=42,     # seed biar pembagiannya sama tiap run
)

model = make_pipeline(                  # rangkai 2 langkah jadi satu model:
    StandardScaler(),                   # 1. samakan skala semua fitur (rata-rata 0, sebaran 1)
    LogisticRegression(max_iter=1000),  # 2. baru masuk ke model klasifikasi
)
model.fit(X_train, y_train)    # TRAINING: scaler belajar skala dari data latihan, lalu model belajar pola
pred = model.predict(X_test)   # INFERENCE: data ujian diskala pakai ukuran yang sama, lalu ditebak (hasil 0/1)


# ============================================================
# 3. EVALUASI
# ============================================================
dummy = np.zeros(len(y_test))   # "satpam males": tebak 0 (aman) untuk semua transaksi

print("Akurasi satpam males:", accuracy_score(y_test, dummy))   # akurasi satpam males (jawaban asli dulu, baru tebakan)
print("Akurasi model:", accuracy_score(y_test, pred))           # akurasi model beneran

print(classification_report(y_test, pred, digits=3))   # tabel precision, recall, f1, support per kelas (0=aman, 1=fraud)
# precision : dari yang dituduh fraud, berapa % yang beneran fraud (seberapa sering salah tuduh)
# recall    : dari semua fraud, berapa % yang ketangkap (seberapa banyak yang lolos)
# f1-score  : gabungan precision & recall
# support   : jumlah data asli di kelas itu


# ============================================================
# 4. RECALL MANUAL (buat verifikasi rumus)
# ============================================================
ketangkap = ((pred == 1) & (y_test == 1)).sum()   # model tebak fraud DAN memang fraud -> dihitung (True=1, False=0)
total_fraud = (y_test == 1).sum()                 # total fraud asli di data ujian

print("Fraud ketangkap:", ketangkap, "dari", total_fraud)
print("Recall manual:", ketangkap / total_fraud)  # harusnya sama dengan recall baris "1" di classification_report