import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_train_ratings, load_test_ratings, load_movies

def clean_similarity(sim):
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, 0, 1)
    return sim

def user_similarity(matrix):
    matrix_adj = matrix - matrix.mean(axis=1).values.reshape(-1, 1)
    sim = cosine_similarity(matrix_adj)
    sim = clean_similarity(sim)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

def predict_user(user_id, movie_id, user_sim, train_matrix, top_k=30):
    if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
        return np.nan

    sim_scores = user_sim.loc[user_id].values
    ratings = train_matrix[movie_id].values
    mask = ratings > 0
    if mask.sum() == 0:
        return np.nan

    similar_users_idx = np.argsort(sim_scores[mask])[-top_k:]
    selected_users = np.where(mask)[0][similar_users_idx]

    weights = sim_scores[selected_users]
    rated_values = ratings[selected_users]

    if weights.sum() == 0:
        return np.nan

    return np.dot(weights, rated_values) / weights.sum()

def get_top_n_recommendations(user_id, user_sim, train_matrix, movie_titles, n=10, top_k_neighbors=30):
    if user_id not in train_matrix.index:
        return pd.DataFrame()

    unrated_movies = train_matrix.loc[user_id][train_matrix.loc[user_id] == 0].index
    predictions = []

    for movie_id in unrated_movies:
        pred = predict_user(user_id, movie_id, user_sim, train_matrix, top_k=top_k_neighbors)
        if not np.isnan(pred):
            title = movie_titles.get(movie_id, f"MovieID {movie_id}")
            predictions.append((movie_id, pred, title))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(predictions[:n], columns=['MovieID', 'Predicted_Rating', 'Title'])

def show_recommendations(train_ratings, test_ratings, top_n=10, top_k_neighbors=30, max_users=20):
    print(f"USER-BASED COLLABORATIVE FILTERING - TOP {top_n} RECOMMENDATIONS".center(85))

    train_matrix = train_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    user_sim = user_similarity(train_matrix)

    movies_df = load_movies()
    movie_titles = movies_df.set_index('ID')['Title'].to_dict()
    print(f"Loaded {len(movie_titles)} movie titles")

    users_to_show = test_ratings['UserID'].unique()[:max_users]

    for user_id in users_to_show:
        if user_id not in train_matrix.index:
            continue

        recs = get_top_n_recommendations(
            user_id=user_id,
            user_sim=user_sim,
            train_matrix=train_matrix,
            movie_titles=movie_titles,
            n=top_n,
            top_k_neighbors=top_k_neighbors
        )

        print(f"User {user_id} | {len(recs)} recommendations:")
        if recs.empty:
            print("   No recommendations available\n")
            continue

        for idx, row in recs.iterrows():
            print(f"   {idx+1:2d}. {int(row['MovieID']):4d} | {row['Predicted_Rating']:.3f} | {row['Title']}")

        print()

    print("="*85)
    print(f"Completed! Showed recommendations for {min(len(users_to_show), max_users)} users")
    print("="*85)

if __name__ == "__main__":
    print("Loading data...")
    train_ratings = load_train_ratings()
    test_ratings = load_test_ratings()
    print(f"Train: {len(train_ratings):,} ratings | Test: {len(test_ratings):,} ratings\n")

    show_recommendations(
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        top_n=10,
        top_k_neighbors=10,
        max_users=10
    )