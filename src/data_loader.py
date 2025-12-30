import pandas as pd
from pathlib import Path
from src.db import load_unprocessed_events

DATA_DIR = Path("./data")
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = DATA_DIR / "vector"

LINK_PATH = PROCESSED_DIR / "links2.csv"

_tmdb_to_ml = None

def load_tmdb_to_movielens_map():
    global _tmdb_to_ml

    if _tmdb_to_ml is None:
        df = pd.read_csv(LINK_PATH)

        print("[DEBUG] links.csv columns:", df.columns.tolist())
        print("[DEBUG] tmdbId sample:", df["tmdbId"].dropna().head(10).tolist())

        # đảm bảo kiểu dữ liệu
        df = df.dropna(subset=["tmdbId"])
        df["tmdbId"] = df["tmdbId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)

        _tmdb_to_ml = dict(zip(df["tmdbId"], df["movieId"]))

    return _tmdb_to_ml
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
    # file_path = PROCESSED_DIR / "user_actions.csv"
    # return pd.read_csv(file_path)
    actions = [
        {"ID": "movie_click", "Name": "movie_click", "Weight": 1.0},
        {"ID": "viewed_movie",  "Name": "viewed_movie", "Weight": 2.0},
        {"ID": "remove_from_watchlist", "Name": "remove_from_watchlist",  "Weight": -2},
        {"ID": "add_to_watchlist",  "Name": "add_to_watchlist",   "Weight": 3.0},
    ]

    return pd.DataFrame(actions)

# def load_user_profiles():
#     # file_path = PROCESSED_DIR / "user_profiles.csv"
#     # return pd.read_csv(file_path)
#      events = load_unprocessed_events()

#     if events.empty:
#         return pd.DataFrame(columns=["UserID", "MovieID", "ActionID"])

#     return pd.DataFrame({
#         "UserID": events["userId"],
#         "MovieID": events["movieId"],
#         "ActionID": events["eventName"]
#     })

# def load_user_profiles():
#     events = load_unprocessed_events()

#     if events.empty:
#         return pd.DataFrame(columns=["UserID", "MovieID", "ActionID", "id"])

#     tmdb_to_ml = load_tmdb_to_movielens_map()

#     # map tmdbId -> movielensId
#     events["MovieID"] = events["movieId"].map(tmdb_to_ml)

#     # loại bỏ event không map được
#     events = events.dropna(subset=["MovieID"])
#     events["MovieID"] = events["MovieID"].astype(int)

#     return pd.DataFrame({
#         "UserID": events["userId"],
#         "MovieID": events["movieId"],
#         "ActionID": events["eventName"],
#         "id": events["id"]
#     })


def load_user_profiles():
    events = load_unprocessed_events()

    if events.empty:
        print("[WARN] No unprocessed events found.")
        return pd.DataFrame(columns=["UserID", "MovieID", "ActionID", "id"])

    print(f"[INFO] Loaded {len(events)} raw events")

    print("[DEBUG] events.movieId dtype:", events["movieId"].dtype)
    print("[DEBUG] events.movieId sample:", events["movieId"].head(5).tolist())


    tmdb_to_ml = load_tmdb_to_movielens_map()
    print(f"[INFO] Loaded TMDB → MovieLens mapping: {len(tmdb_to_ml)} entries")

    # map tmdbId -> movielensId
    events["MovieID"] = events["movieId"].map(tmdb_to_ml)

    total_events = len(events)
    mapped_events = events["MovieID"].notna().sum()
    dropped_events = total_events - mapped_events

    print("[INFO] Mapping result:")
    print(f"  ✔ Mapped successfully : {mapped_events}")
    print(f"  ✘ Dropped (no mapping): {dropped_events}")

    # in thử một vài mapping mẫu
    print("[INFO] Sample TMDB → MovieLens mapping:")
    sample = events.loc[events["MovieID"].notna(), ["movieId", "MovieID"]].head(5)
    for _, row in sample.iterrows():
        print(f"  TMDB {row['movieId']} → ML {int(row['MovieID'])}")

    # loại bỏ event không map được
    events = events.dropna(subset=["MovieID"])
    events["MovieID"] = events["MovieID"].astype(int)

    result = pd.DataFrame({
        "UserID": events["userId"],
        "MovieID": events["MovieID"],   # ✅ dùng movielensId
        "ActionID": events["eventName"],
        "id": events["id"]
    })

    print(f"[INFO] Final user profile rows: {len(result)}")

    return result

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
    file_path = VECTOR_DIR / "user_profile_vectors_db.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy {file_path}")

def load_train_ratings(version: str = "80"):
    file_path = PROCESSED_DIR / f"train_ratings_{version}.csv"
    return pd.read_csv(file_path)

def load_test_ratings(version: str = "20"):
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