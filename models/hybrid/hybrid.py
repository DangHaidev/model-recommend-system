import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_movie_vectors, load_train_ratings
from models.collaborative.cf_user_based import predict_user
from models.collaborative.cf_item_based import predict_item

import models.content_based.cb_movie as cb

movie_vecs_df = load_movie_vectors()
movie_ids = movie_vecs_df["ID"].values

train_df = load_train_ratings()
user_item_matrix = train_df.pivot(index="UserID", columns="MovieID", values="Rating").fillna(0)

movies_meta = pd.read_csv(Path("./data/processed/movies.csv"))
id_to_title = dict(zip(movies_meta["ID"], movies_meta["Title"]))

def build_user_similarity(train_matrix):
    matrix_adj = train_matrix - train_matrix.mean(axis=1).values.reshape(-1, 1)
    sim = cosine_similarity(matrix_adj)
    sim = np.nan_to_num(sim)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=train_matrix.index, columns=train_matrix.index)


def build_item_similarity(train_matrix):
    matrix_adj = train_matrix - train_matrix.mean(axis=0)
    sim = cosine_similarity(matrix_adj.T)
    sim = np.nan_to_num(sim)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)


user_sim = build_user_similarity(user_item_matrix)
item_sim = build_item_similarity(user_item_matrix)

USER_WEIGHT = 4.5 / (4.5 + 5.5)
ITEM_WEIGHT = 5.5 / (4.5 + 5.5)
ALPHA = 0.5


def recommend_hybrid(user_id, top_n=10):
    if user_id not in user_item_matrix.index:
        return []

    user_history = user_item_matrix.loc[user_id]
    user_history = user_history[user_history > 0].to_dict()

    cb_scores = np.zeros(len(cb.movie_ids))
    for mid, rating in user_history.items():
        idx_list = np.where(cb.movie_ids == mid)[0]
        if len(idx_list) == 0:
            continue
        idx = idx_list[0]
        cb_scores += cb.similarity_matrix[idx] * rating

    if np.max(cb_scores) > 0:
        cb_scores = normalize(cb_scores.reshape(1, -1))[0]

    cf_scores = np.zeros(len(cb.movie_ids))
    for i, mid in enumerate(cb.movie_ids):
        if mid not in user_history:
            a = predict_user(user_id, mid, user_sim, user_item_matrix)
            b = predict_item(user_id, mid, item_sim, user_item_matrix)

            a = 0 if np.isnan(a) else a
            b = 0 if np.isnan(b) else b

            cf_scores[i] = USER_WEIGHT * a + ITEM_WEIGHT * b

    if np.max(cf_scores) > 0:
        cf_scores = normalize(cf_scores.reshape(1, -1))[0]

    hybrid_score = ALPHA * cb_scores + (1 - ALPHA) * cf_scores

    top_idx = np.argsort(hybrid_score)[::-1][:top_n]
    return [cb.movie_ids[i] for i in top_idx]

if __name__ == "__main__":
    print("Running hybrid recommender test...\n")

    user_id = 1
    result = recommend_hybrid(user_id, top_n=10)

    for mid in result:
        print(f"- {mid} : {id_to_title.get(mid, 'Unknown')}")
