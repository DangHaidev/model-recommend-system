import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import sys
import ast
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader import (
    load_user_profiles,
    load_user_actions,
    load_movie_vectors
)

OUTPUT_PATH = Path("./data/vector/user_profile_vectors.csv")


def compute_user_profile_vectors(
    user_profiles_df: pd.DataFrame,
    user_actions_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    save_to_csv: bool = True,
    output_path: Path = None
):
    df = user_profiles_df.merge(
        user_actions_df,
        left_on="ActionID",
        right_on="ID",
        how="left"
    )

    movies_df["movieVector"] = movies_df["movieVector"].apply(ast.literal_eval)

    movie_vector_dict = dict(zip(movies_df["ID"], movies_df["movieVector"]))

    user_vectors = []

    print("Calculating user vectors for all users...")
    for uid, user_data in tqdm(df.groupby("UserID"), total=df["UserID"].nunique()):
        weighted_vecs = []
        weights = []

        for _, row in user_data.iterrows():
            movie_vec = movie_vector_dict.get(row["MovieID"])
            if movie_vec is None:
                continue

            movie_vec = np.array(movie_vec)
            w = row["Weight"]

            weighted_vecs.append(movie_vec * w)
            weights.append(w)

        if len(weighted_vecs) == 0:
            continue

        user_vec = np.sum(weighted_vecs, axis=0) / sum(weights)
        user_vec = normalize(user_vec.reshape(1, -1))[0]

        user_vectors.append({
            "UserID": uid,
            "userVector": user_vec.tolist()
        })

    result = pd.DataFrame(user_vectors)

    if save_to_csv and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"✔ File saved: {output_path}")

    return result


if __name__ == "__main__":
    movies_df = load_movie_vectors()  
    user_profiles_df = load_user_profiles()
    user_actions_df = load_user_actions()

    output = compute_user_profile_vectors(
        user_profiles_df,
        user_actions_df,
        movies_df,
        output_path=OUTPUT_PATH
    )

    print(output.head())
