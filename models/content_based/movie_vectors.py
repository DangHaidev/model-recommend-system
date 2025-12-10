import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_movies
OUTPUT_PATH = Path("./data/vector/movie_vectors.csv")

def compute_movie_vectors(
    movies_df: pd.DataFrame, 
    save_to_csv: bool = True, 
    output_path: Path = None
):
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded!")

    movies = movies_df.copy()

    print("Encoding features...")

    title_vec    = model.encode(movies["Title"].fillna("").tolist())
    genres_vec   = model.encode(movies["Genres"].fillna("").tolist())
    overview_vec = model.encode(movies["Overview"].fillna("").tolist())
    keyword_vec  = model.encode(movies["Keyword"].fillna("").tolist())
    network_vec  = model.encode(movies["Network"].fillna("").tolist())

    print("Encoding done!")

    title_vec    = normalize(title_vec)
    genres_vec   = normalize(genres_vec)
    overview_vec = normalize(overview_vec)
    keyword_vec  = normalize(keyword_vec)
    network_vec  = normalize(network_vec)

    weights = {
        "title": 0.41,
        "genres": 0.19,
        "keyword": 0.12,
        "overview": 0.10,
        "network": 0.18
    }

    print("\nUsing weights:")
    for k, v in weights.items():
        print(f"{k}: {v}")

    final_movie_vec = (
        title_vec    * weights["title"] +
        genres_vec   * weights["genres"] +
        keyword_vec  * weights["keyword"] +
        overview_vec * weights["overview"] +
        network_vec  * weights["network"]
    )

    final_movie_vec = normalize(final_movie_vec)
    movies["movieVector"] = final_movie_vec.tolist()
    result = movies[["ID", "Title", "movieVector"]].copy()

    if save_to_csv and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print("File saved:", output_path)

    return result


if __name__ == "__main__":
    movies_df = load_movies()
    output = compute_movie_vectors(
        movies_df,
        output_path=OUTPUT_PATH
    )
    print(output.head())
