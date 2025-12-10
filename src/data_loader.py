import pandas as pd
from pathlib import Path

DATA_DIR = Path("./data")
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = DATA_DIR / "vector"
def load_movies():
    file_path = PROCESSED_DIR / "movies.csv"
    return pd.read_csv(file_path)

def load_links():
    file_path = PROCESSED_DIR / "links.csv"
    return pd.read_csv(file_path)

def load_ratings():
    file_path = PROCESSED_DIR / "ratings.csv"
    return pd.read_csv(file_path)

def load_feature_correlation():
    file_path = PROCESSED_DIR / "feature_correlation.csv"
    return pd.read_csv(file_path)

def load_user_actions():
    file_path = PROCESSED_DIR / "user_actions.csv"
    return pd.read_csv(file_path)

def load_user_profiles():
    file_path = PROCESSED_DIR / "user_profiles.csv"
    return pd.read_csv(file_path)

def load_users():
    file_path = PROCESSED_DIR / "users.csv"
    return pd.read_csv(file_path)

def load_movie_vectors():
    file_path = VECTOR_DIR / "movie_vectors.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy {file_path}")

def load_user_profile_vectors():
    file_path = VECTOR_DIR / "user_profile_vectors.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy {file_path}")

def load_train_ratings(version: str = "8.0"):
    file_path = PROCESSED_DIR / f"train_ratings_{version}.csv"
    return pd.read_csv(file_path)

def load_test_ratings(version: str = "2.0"):
    file_path = PROCESSED_DIR / f"test_ratings_{version}.csv"
    return pd.read_csv(file_path)

def check_all_files():
    required_files = [
        "movies.csv", "links.csv", "ratings.csv",
        "user_actions.csv", "user_profiles.csv", "users.csv",
        "feature_correlation.csv"
    ]
    missing = [f for f in required_files if not (PROCESSED_DIR / f).exists()]
    if missing:
        print("Thiếu các file sau:", missing)
    else:
        print("Tất cả file cần thiết đều đã có sẵn!")

if __name__ == "__main__":
    check_all_files()
    
    movies = load_movies()
    ratings = load_ratings()
    user_actions = load_user_actions()
    
    print(f"movies: {movies.shape}")
    print(f"ratings: {ratings.shape}")
    print(f"user_actions: {user_actions.shape}")