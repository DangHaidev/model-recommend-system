import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hybrid.hybrid import recommend_hybrid
from models.collaborative.cf_user_based import predict_user
from models.collaborative.cf_item_based import predict_item
from models.content_based.cb_movie import movie_ids, similarity_matrix
from src.data_loader import load_train_ratings

# -------------------- Generic trainer --------------------
def train_model(recommend_func, users, top_n=10, output_path="./output/result/recs.pkl"):
    results = {}
    for u in users:
        try:
            recs = recommend_func(u, top_n=top_n)
            results[u] = recs
        except Exception:
            results[u] = []
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    return results

# -------------------- CB wrapper using user history --------------------
def recommend_cb_user(user_id, top_n=10):
    global user_item_matrix, movie_ids, similarity_matrix

    if user_id not in user_item_matrix.index:
        return []

    # Lấy lịch sử rating của user
    user_history = user_item_matrix.loc[user_id]
    user_history = user_history[user_history > 0].to_dict()  # {movie_id: rating}

    if not user_history:
        return []

    # Tính CB scores dựa trên similarity
    cb_scores = np.zeros(len(movie_ids))
    for mid, rating in user_history.items():
        if mid not in movie_ids:
            continue
        idx = np.where(movie_ids == mid)[0][0]
        cb_scores += similarity_matrix[idx] * rating

    # Chuẩn hóa
    if np.max(cb_scores) > 0:
        cb_scores = normalize(cb_scores.reshape(1, -1))[0]

    # Loại bỏ phim đã xem
    recommended_movies = [movie_ids[i] for i in np.argsort(cb_scores)[::-1] if movie_ids[i] not in user_history]
    
    return recommended_movies[:top_n]

# -------------------- CF wrappers --------------------
def recommend_cf_user(user_id, top_n=10):
    global user_item_matrix
    recs = []
    for mid in user_item_matrix.columns:
        score = predict_user_based(user_id, mid)
        if np.isnan(score):
            score = 0
        recs.append((mid, score))
    recs.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in recs[:top_n]]

def recommend_cf_item(user_id, top_n=10):
    global user_item_matrix
    recs = []
    for mid in user_item_matrix.columns:
        score = predict_item_based(user_id, mid)
        if np.isnan(score):
            score = 0
        recs.append((mid, score))
    recs.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in recs[:top_n]]

# -------------------- Main --------------------
if __name__ == "__main__":
    limit = 100       # số user muốn train
    top_n = 10

    ratings = load_train_ratings()
    users = ratings["UserID"].unique()[:limit]

    # --- load user_item_matrix cho CF và CB ---
    user_item_matrix = ratings.pivot(index="UserID", columns="MovieID", values="Rating").fillna(0)

    # --- Train and save CB ---
    print("Training Content-Based...")
    train_model(recommend_cb_user, users, top_n=top_n, output_path="./output/result/cb_recommendations.pkl")

    # --- Train and save CF User-Based ---
    print("Training CF User-Based...")
    train_model(recommend_cf_user, users, top_n=top_n, output_path="./output/result/cf_user_recommendations.pkl")

    # --- Train and save CF Item-Based ---
    print("Training CF Item-Based...")
    train_model(recommend_cf_item, users, top_n=top_n, output_path="./output/result/cf_item_recommendations.pkl")

    # --- Train and save Hybrid ---
    print("Training Hybrid...")
    train_model(recommend_hybrid, users, top_n=top_n, output_path="./output/result/hybrid_recommendations.pkl")

    print("All models trained and saved!")
