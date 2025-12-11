import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_movie_vectors, load_train_ratings, load_test_ratings

movie_vecs_df = load_movie_vectors()
movie_vecs_df['movieVector'] = movie_vecs_df['movieVector'].apply(lambda x: np.array(eval(x)))
movie_vectors = np.stack(movie_vecs_df['movieVector'].values)
movie_ids = movie_vecs_df['ID'].values
movieid_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

movies_meta = pd.read_csv(Path("./data/processed/movies.csv"))
id_to_title = dict(zip(movies_meta["ID"], movies_meta["Title"]))

train_df = load_train_ratings()
user_item_matrix = train_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

cb_sim_matrix = cosine_similarity(movie_vectors)


def clean_similarity(sim):
    sim = np.nan_to_num(sim, nan=0.0)
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0, 1)


user_mean = user_item_matrix.mean(axis=1).values.reshape(-1, 1)
user_adj_matrix = user_item_matrix - user_mean
user_sim = clean_similarity(cosine_similarity(user_adj_matrix))
user_sim = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

item_mean = user_item_matrix.mean(axis=1).values.reshape(-1, 1)
item_adj_matrix = user_item_matrix - item_mean
item_sim = clean_similarity(cosine_similarity(item_adj_matrix.T))
item_sim = pd.DataFrame(item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

USER_WEIGHT = 4.5 / (4.5 + 5.5)
ITEM_WEIGHT = 5.5 / (4.5 + 5.5)
ALPHA = 0.5


def predict_user_based(user_id, movie_id, top_k=50):
    if user_id not in user_item_matrix.index or movie_id not in user_item_matrix.columns:
        return np.nan

    sims = user_sim.loc[user_id].values
    ratings = user_item_matrix[movie_id].values

    mask = ratings > 0
    if mask.sum() == 0:
        return np.nan

    sims = sims[mask]
    ratings = ratings[mask]

    if len(sims) > top_k:
        idx = np.argsort(sims)[-top_k:]
        sims = sims[idx]
        ratings = ratings[idx]

    if sims.sum() == 0:
        return np.nan

    return np.dot(sims, ratings) / sims.sum()


def predict_item_based(user_id, movie_id, top_k=50):
    if user_id not in user_item_matrix.index or movie_id not in user_item_matrix.columns:
        return np.nan

    sims = item_sim[movie_id].values
    ratings = user_item_matrix.loc[user_id].values

    mask = ratings > 0
    if mask.sum() == 0:
        return np.nan

    sims = sims[mask]
    ratings = ratings[mask]

    if len(sims) > top_k:
        idx = np.argsort(sims)[-top_k:]
        sims = sims[idx]
        ratings = ratings[idx]

    if sims.sum() == 0:
        return np.nan

    return np.dot(sims, ratings) / sims.sum()


def recommend_hybrid(user_id, top_n=10):
    if user_id in user_item_matrix.index:
        user_history = user_item_matrix.loc[user_id]
        user_history = user_history[user_history > 0].to_dict()
    else:
        user_history = {}

    cb_scores = np.zeros(len(movie_ids))
    for mid, rating in user_history.items():
        if mid not in movieid_to_index:
            continue
        idx = movieid_to_index[mid]
        cb_scores += cb_sim_matrix[idx] * rating

    if np.max(cb_scores) > 0:
        cb_scores = normalize(cb_scores.reshape(1, -1))[0]

    cf_scores = np.zeros(len(movie_ids))
    for i, mid in enumerate(movie_ids):
        if mid in user_history:
            cf_scores[i] = 0
        else:
            a = predict_user_based(user_id, mid)
            b = predict_item_based(user_id, mid)

            a = 0 if np.isnan(a) else a
            b = 0 if np.isnan(b) else b

            cf_scores[i] = USER_WEIGHT * a + ITEM_WEIGHT * b

    if np.max(cf_scores) > 0:
        cf_scores = normalize(cf_scores.reshape(1, -1))[0]

    hybrid_scores = ALPHA * cb_scores + (1 - ALPHA) * cf_scores

    top_idx = np.argsort(hybrid_scores)[::-1][:top_n]
    return [movie_ids[i] for i in top_idx]


if __name__ == "__main__":
    print("Running hybrid recommender test...\n")

    user_id = 1
    top10 = recommend_hybrid(user_id, top_n=10)

    print(f"Top 10 gợi ý cho user {user_id}:\n")
    for mid in top10:
        title = id_to_title.get(mid, "Không tìm thấy tên phim")
        print(f"- {mid} : {title}")
