import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_movie_vectors

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

def recommend_similar_movies(movie_id, top_n=10):
    idx = np.where(movie_ids == movie_id)[0][0]
    sims = similarity_matrix[idx].copy()
    sims[idx] = -np.inf 
    top_idx = np.argsort(sims)[::-1][:top_n]
    top_movie_ids = movie_ids[top_idx]
    top_titles = movies_df.set_index('ID').loc[top_movie_ids, 'Title'].values
    return pd.DataFrame({'ID': top_movie_ids, 'Title': top_titles, 'sim': sims[top_idx]})

if __name__ == "__main__":
    movie_id = 1
    top_movies = recommend_similar_movies(movie_id, top_n=10)
    print(f"Top 10 movies similar to movie {movie_id} using Manhattan distance:")
    print(top_movies)
