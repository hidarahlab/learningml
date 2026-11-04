import numpy as np, pandas as pd

# # ============================================================
# # 1. BIKIN DATA SIMULASI
# # ============================================================
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