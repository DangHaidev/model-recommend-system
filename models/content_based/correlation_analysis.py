import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_movies

def compute_feature_correlations(movies_df: pd.DataFrame):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    movies_df = movies_df.fillna("")

    required_columns = ["Title", "Genres", "Keyword", "Overview", "Network"]
    missing = [col for col in required_columns if col not in movies_df.columns]
    if missing:
        raise ValueError(f"Missing columns in movies_df: {missing}")

    title_vec    = model.encode(movies_df["Title"].tolist(), show_progress_bar=True)
    genres_vec   = model.encode(movies_df["Genres"].tolist(), show_progress_bar=True)
    keyword_vec  = model.encode(movies_df["Keyword"].tolist(), show_progress_bar=True)
    overview_vec = model.encode(movies_df["Overview"].tolist(), show_progress_bar=True)
    network_vec  = model.encode(movies_df["Network"].tolist(), show_progress_bar=True)

    corr_genres   = cosine_similarity(title_vec, genres_vec).mean()
    corr_keyword  = cosine_similarity(title_vec, keyword_vec).mean()
    corr_overview = cosine_similarity(title_vec, overview_vec).mean()
    corr_network  = cosine_similarity(title_vec, network_vec).mean()

    result = pd.DataFrame({
        "Feature": ["Genres", "Keyword", "Overview", "Network"],
        "Correlation_with_Title": [
            corr_genres,
            corr_keyword,
            corr_overview,
            corr_network,
        ]
    })

    return result


if __name__ == "__main__":
    movies_df = load_movies()
    corr_table = compute_feature_correlations(movies_df)
    print("Feature Correlation Table")
    print(corr_table)
    corr_table.to_csv(Path("./data/processed/feature_correlation.csv"), index=False)    