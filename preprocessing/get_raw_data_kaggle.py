import os
import shutil
import kagglehub

OUTPUT_CSV = "telco_churn_raw.csv"

# 1. Mengambil lokasi script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Naik 1 tingkat dari lokasi script (Parent Directory)
ROOT_FOLDER = os.path.dirname(SCRIPT_DIR)

# 3. Menentukan direktori target output (naik 1 tingkat -> folder preprocessing_new)
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "preprocessing_new")

# 4. Pastikan folder 'preprocessing_new' dibuat jika belum ada
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 5. Path lengkap tempat file CSV akan disimpan
target_file_path = os.path.join(OUTPUT_FOLDER, OUTPUT_CSV)

# 6. Unduh dataset via kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# 7. Cari dan salin file CSV ke lokasi tujuan
downloaded_files = [f for f in os.listdir(path) if f.endswith(".csv")]

if downloaded_files:
    source_file = os.path.join(path, downloaded_files[0])

    # Salin file ke target
    shutil.copy(source_file, target_file_path)
    print(f"File berhasil disimpan di: {os.path.abspath(target_file_path)}")
else:
    print("Tidak ditemukan file CSV di direktori unduhan.")