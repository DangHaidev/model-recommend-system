import pandas as pd
import os

DATA_DIR = "data/processed/"

# ----------------------------------------------------------
# 1. HÀM ĐỌC FILES
# ----------------------------------------------------------
def load_movies(file_path=None):
    """Đọc dữ liệu phim sau tiền xử lý"""
    path = file_path or os.path.join(DATA_DIR, "movies_after_preprocessing.csv")
    movies = pd.read_csv(path)
    print(f"✅ Loaded {len(movies)} movies")
    return movies


def load_ratings(file_path=None):
    """Đọc dữ liệu ratings"""
    path = file_path or os.path.join(DATA_DIR, "ratings_after_preprocessing.csv")
    ratings = pd.read_csv(path)
    print(f"✅ Loaded {len(ratings)} ratings")
    return ratings


def load_users(file_path=None):
    """Đọc dữ liệu người dùng"""
    path = file_path or os.path.join(DATA_DIR, "users_after_preprocessing.csv")
    users = pd.read_csv(path)
    print(f"✅ Loaded {len(users)} users")
    return users
