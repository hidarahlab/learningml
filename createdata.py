import numpy as np, pandas as pd
from datetime import datetime

# ============================================================
# 0. PENGATURAN SEED
# ============================================================
SEED = None   # angka   -> data selalu sama tiap run (buat bandingin eksperimen)
            # None    -> data acak tiap run, seed-nya di-print biar bisa diulang

if SEED is None:                                          # kalau mode acak...
    SEED = int(np.random.SeedSequence().entropy % 10_000) # ...pilih seed acak 0-9999
print("Seed yang dipakai:", SEED)                         # SELALU catat seed-nya

rng = np.random.default_rng(SEED)   # generator angka acak berdasarkan seed di atas


# ============================================================
# 1. BIKIN DATA SIMULASI
# ============================================================
n = 10_000   # jumlah transaksi (_ cuma pemisah biar gampang dibaca)

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

df["fraud"] = (skor + rng.normal(0, 1, n) > 5.5).astype(int)  # tambah noise acak, skor > 5.5 = fraud (1), selain itu aman (0)

print(df.head())                              # tampilkan 5 baris pertama
print("Jumlah baris & kolom:", df.shape)      # (baris, kolom)
print("Rasio fraud:", df.fraud.mean())        # mean dari kolom 0/1 = proporsi fraud


# ============================================================
# 2. SIMPAN KE CSV
# ============================================================
waktu = datetime.now().strftime("%Y%m%d_%H%M%S")    # format: 20261105_081830 (tahun bulan tgl _ jam menit detik)
nama_file = f"data_transaksi_{waktu}.csv"           # f-string: {waktu} diganti isi variabel waktu
df.to_csv(nama_file, index=False)                   # index=False: kolom 0,1,2,... nggak ikut disimpan
print("Tersimpan ke", nama_file)