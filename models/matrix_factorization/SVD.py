import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_train_ratings, load_movies

def svd_top10(train_ratings, user_id, n_factors=20, top_n=10):
    train_matrix = train_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    matrix_values = train_matrix.values

    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(matrix_values)
    item_factors = svd.components_.T
    reconstructed = np.dot(user_factors, item_factors.T)
    reconstructed_df = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)

    preds = []
    for movie_id in train_matrix.columns:
        if train_matrix.loc[user_id, movie_id] == 0:
            preds.append((movie_id, reconstructed_df.loc[user_id, movie_id]))

    top_movies = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

if __name__ == "__main__":
    train_ratings = load_train_ratings("80")
    movies = load_movies()
    user_id = 1
    top10 = svd_top10(train_ratings, user_id, n_factors=20, top_n=10)

    print(f"Top 10 movies recommended for user {user_id}:")
    for rank, (mid, score) in enumerate(top10, 1):
        title = movies.loc[movies['ID'] == mid, 'Title'].values[0]
        print(f"{rank}. {title} (MovieID {mid}) - Score: {score:.2f}")
