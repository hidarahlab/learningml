# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message=".*encountered in matmul",
#     category=RuntimeWarning
# )

# ============================================================
# IMPORT LIBRARY
# ============================================================
# import glob                                                     # library cari nama file
# import numpy as np, pandas as pd                                # library numpy = array matematik, pandas = table (Data Frame)
# from sklearn.model_selection import train_test_split            # library alat buat bagi data train/test
# from sklearn.linear_model import LogisticRegression             # library algoritma model klasifikasi
# from sklearn.metrics import accuracy_score                      # library alat buat mengukur nilai model
# from sklearn.metrics import classification_report               # library alat buat mengukur nilai moedel
# from sklearn.preprocessing import StandardScaler                # library alat buat menyamakan skala fitur
# from sklearn.pipeline import make_pipeline                      # library alat buat merangkat beberapa langkah jadi satu

# ============================================================
# 1. BACA DATA DARI CSV (otomatis ambil file paling baru)
# ============================================================
# daftar_file = sorted(glob.glob("data_transaksi_*.csv"))             # cari semua file yang cocok

# if not daftar_file:
#     raise SystemExit("data kosong")                                 # jika data kosong return print "data kosong"

# nama_file = daftar_file[-1]                                         # [-1] = elemen paling akhir = file paling baru
# print("Membaca: ", nama_file)                                       # tampilkan nama file

# df = pd.read_csv(nama_file)                                         # baca CSV jadi DataFrame (kebalikan dari to_csv)
# print(df.head())                                                    # tampilkan 5 baris pertama
# print("Jumlah baris & kolom: ", df.shape)                           # baris, kolom
# print("Rasio fraud: ", df.fraud.mean())                             # mean dari kolom 0/1 proporsi fraud

# ============================================================
# 2. TRAINING
# ============================================================
# X = df.drop(columns="fraud")                                        # FITUR: semua kolom kecuali fraud (yang dilihat model)
# y = df.fraud                                                        # LABEL: kolom fraud saja(kunci jawaban)

# X_train, X_test, y_train, y_test = train_test_split (               # bagi data jadi train & test
#         X,y,
#         test_size=0.2,                                              # 20% buat ujian, 80% buat belajar (8000)
#         stratify=y,                                                 # jaga rasio fraud sama di train dan test (~1,5%)
#         random_state=42                                             # seed pembagian data, tetap 42 biar eksperimen adil
# )

# model = make_pipeline(                                              # rangkai 2 lamgkah jadi satu model
#         StandardScaler(),                                           # 1. samakan skala semua fitur (rata-rata 0, sebaran 1)
#         LogisticRegression(max_iter=1000)                           # 2. baru masuk ke model klasifikasi
# )
# model.fit(X_train, y_train)                                         # TRAINING: scaler belajar skala dari data latihan, lalu model belajar pola
# pred = model.predict(X_test)                                        # INFERENCE: data ujian diskala pakai ukuran yang sama, lalu di tebak(hasil 0/1)

# ============================================================
# 3. EVALUASI
# ============================================================
# dummy = np.zeros(len(y_test))                                       # "Satpam males" : tebak 0 (aman) untuk semua transaksi

# print("Akurasi satpam males:", accuracy_score(y_test, dummy))       # akurasi satpam (jawaban asli dulu, baru tebakan)
# print("Akurasi model: ", accuracy_score(y_test, pred))              # akurasi model beneran

# print(classification_report(y_test, pred, digits=3))                # table precsion, recall, f1, support per kelas (0=aman, 1=fraud)
# # precision : dari yang dituduh fraud, berapa % yang beneran fraud (seberapa sering salah tuduh)
# # recall    : dari semua fraud, berapa % yang ketangkap (seberapa banyak yang lolos)
# # f1-score  : gabungan precision & recall
# # support   : jumlah data asli di kelas itu
dummy = np.zeros(len(y_test))                                   # "Satpam males" : tebak 0 (aman) untuk semua transaksi

print("Akurasi satpam males: ", accuracy_score)

# ============================================================
# 4. RECALL MANUAL (buat verifikasi rumus)
# ============================================================
# ketangkap = ((pred == 1) & (y_test == 1)).sum()                 # model tebak fraud dan memang fraud -> dihitung (True=1, False=0)
# total_fraud = (y_test == 1).sum()                               # total fraud asli di data ujian

# print("Fraud ketangkap: ", ketangkap, "dari ", total_fraud)     # perhitungan fraud ketangkap
# print("Recall manual: ", ketangkap / total_fraud)               # perhitungan recall manual

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning
)

# ============================================================
# IMPORT LIBRARY
# ============================================================
import glob                                                     # libary cari nama file
import numpy as np, pandas as pd                                # library numpy = array matematik, pandas = table (Data Frame)
from sklearn.model_selection import train_test_split            # library alat buat bagi data train/test
from sklearn.linear_model import LogisticRegression             # library algoritma model klasifikasi
from sklearn.metrics import accuracy_score                      # library alat buat mengukur nilai model
from sklearn.metrics import classification_report               # library alat buat mengukur nilai model
from sklearn.preprocessing import StandardScaler                # library alat buat menyamakan skala fitur
from sklearn.pipeline import make_pipeline                      # library alat buat merangkat beberapa langkah jadi satu

# ============================================================
# 1. BACA DATA DARI CSV (otomatis ambil file paling baru)
# ============================================================
daftar_file = sorted(glob.glob("data_transaksi_*.csv"))             # cari semua file yang cocok

if not daftar_file:
    raise SystemExit("data kosong")                                 # jika data kosong return print "data kosong"

nama_file = daftar_file = daftar_file[-1]                           # [-1] = elemen paling akhir = file paling baru
print("membaca: ", nama_file)                                       # tampilan nama file

df = pd.read_csv(nama_file)                                         # baca csv jadi DataFrame (kebalikan dari to_csv)
print(df.head())                                                    # tampilkan 5 baris pertama
print("Jumlah baris & kolom: ", df.shape)                           # baris, kolom
print("Rasio fraud: ", df.fraud.mean())                             # mean dari kolom 0/1 proporsi fraud

# ============================================================
# 2. TRAINING
# ============================================================
X = df.drop(columns="fraud")                                    # FITUR: semua kolom kecuali fraud (yang dilihat model)
y = df.fraud                                                    # Label: kolom fraud saja(kunci jawaban)

X_train, X_test, y_train, y_test = train_test_split (           # bagi data jadi train & test
     X,y,                                                       
     test_size=0.2,                                             # rangkai 2 langkah jadi satu model
     stratify=y,                                                # jaga rasio fraud sama di train dan test (~1,5%)
     random_state=42                                            # seed pembagian data, tetap 42 biar eksperimen adil
)

model = make_pipeline(                                          # rangkai 2 langkah jadi satu model
    StandardScaler(),                                           # 1. samakan skala semua fitur (rata-rata 0, sebaran 1)
    LogisticRegression(max_iter=1000)                           # 2. baru masuk ke model klasifikasi
)
model.fit(X_train, y_train)                                     # TRAINING: scaler belajar skala dari data latihan, lalu model belajar pola
prod = model.predict(X_test)                                    # INFERENCE: data ujian diskala pakai ukuran yang sama, lalu di tebak(hasil 0/)

# ============================================================
# 3. EVALUASI
# ============================================================

# ============================================================
# 4. RECALL MANUAL (buat verifikasi rumus)
# ============================================================