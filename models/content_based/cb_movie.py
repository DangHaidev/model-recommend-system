import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_movie_vectors
from src.data_loader import load_movies

movies_df = load_movie_vectors()  
movies_df['movieVector'] = movies_df['movieVector'].apply(lambda x: np.array(eval(x)))

movies_meta = pd.read_csv(Path("./data/processed/movies.csv")) 
movies_meta['GenresSet'] = movies_meta['Genres'].apply(lambda x: set(x.split('|')) if isinstance(x, str) else set())
movieid_to_genres = movies_meta.set_index('ID')['GenresSet'].to_dict()

movie_vecs = np.stack(movies_df['movieVector'].values)
movie_ids = movies_df['ID'].values

# Compute cosine similarity


# Compute Manhattan similarity
def compute_manhattan_similarity(vecs):
    n = vecs.shape[0]
    sims = np.zeros((n, n))
    for i in range(n):
        sims[i] = -np.sum(np.abs(vecs - vecs[i]), axis=1)
    return sims

similarity_matrix = compute_manhattan_similarity(movie_vecs)

def id_to_tmdbid(movie_id):
    movies_df = load_movies()
    row = movies_df.loc[movies_df["ID"] == movie_id]
    if row.empty:
        return None
    tmdb_id = row.iloc[0]["tmdbId"]
    if tmdb_id is None:
        return None
    return int(tmdb_id)


def tmdbid_to_id(tmdb_id):
    movies_df = load_movies()
    row = movies_df.loc[movies_df["tmdbId"] == tmdb_id]
    if row.empty:
        return None
    movie_id = row.iloc[0]["ID"]
    if movie_id is None:
        return None
    return int(movie_id)

def recommend_similar_movies(movie_id_input, top_n=10):
    #convert tmdbid to movie_id
    movie_id = tmdbid_to_id(movie_id_input)
    # tìm index phim trong vector
    idx = np.where(movie_ids == movie_id)[0][0]

    # vector độ tương đồng
    sims = similarity_matrix[idx].copy()
    sims[idx] = -np.inf  # loại bỏ chính nó

    # top N index
    top_idx = np.argsort(sims)[::-1][:top_n]
    top_movie_ids = movie_ids[top_idx]

    # tạo DataFrame kết quả
    result = pd.DataFrame({
        'ID': top_movie_ids,
        'sim': sims[top_idx]
    })

    # merge thông tin meta từ movies_meta
    result = result.merge(
        movies_meta[['ID','Title','tmdbId']],
        on='ID',
        how='left'
    )

    return result



if __name__ == "__main__":
    movie_id = 1
    top_movies = recommend_similar_movies(movie_id, top_n=10)
    print(f"Top 10 movies similar to movie {movie_id} using Manhattan distance:")
    print(top_movies)
