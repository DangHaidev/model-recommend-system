import pandas as pd
import os

DATA_DIR = "data/processed/"

# ----------------------------------------------------------
# 1. HÀM CHUNG ĐỌC FILE CSV
# ----------------------------------------------------------
def read_csv_file(file_name, folder_path=DATA_DIR):
    """Hàm đọc file CSV với đường dẫn linh hoạt"""
    path = os.path.join(folder_path, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded '{file_name}' ({len(df)} rows)")
    return df


# ----------------------------------------------------------
# 2. HÀM RIÊNG TỪNG FILE
# ----------------------------------------------------------
def load_movies(file_path=None):
    """Đọc dữ liệu phim sau tiền xử lý"""
    if file_path:
        return read_csv_file(file_path, folder_path="")
    return read_csv_file("movies_after_preprocessing.csv")


def load_ratings(file_path=None):
    """Đọc dữ liệu ratings"""
    if file_path:
        return read_csv_file(file_path, folder_path="")
    return read_csv_file("ratings_after_preprocessing.csv")


def load_users(file_path=None):
    """Đọc dữ liệu người dùng"""
    if file_path:
        return read_csv_file(file_path, folder_path="")
    return read_csv_file("users_after_preprocessing.csv")
