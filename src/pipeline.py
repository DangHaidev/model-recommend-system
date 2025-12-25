from src.db import load_unprocessed_events, mark_events_processed
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import sys
import ast
from tqdm import tqdm
from src.data_loader import (
    load_user_profiles,
    load_user_actions,
    load_movie_vectors
)

OUTPUT_PATH = Path("./data/vector/user_profile_vectors_db.csv")


def compute_user_profile_vectors(
    user_profiles_df: pd.DataFrame,
    user_actions_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    save_to_csv: bool = True,
    output_path: Path = None
):
    print("🔍 DEBUG: Input shapes")
    print("user_profiles_df:", user_profiles_df.shape)
    print("user_actions_df :", user_actions_df.shape)
    print("movies_df       :", movies_df.shape)

    # ===============================
    # Merge user actions
    # ===============================
    df = user_profiles_df.merge(
        user_actions_df,
        left_on="ActionID",
        right_on="ID",
        how="left"
    )

    # Check missing weight
    missing_weight = df["Weight"].isna().sum()
    if missing_weight > 0:
        print(f"⚠ WARNING: {missing_weight} rows have NaN Weight")

    # ===============================
    # Parse movie vectors
    # ===============================
    def parse_vector(x, movie_id):
        try:
            v = np.array(ast.literal_eval(x), dtype=np.float32)
            if np.isnan(v).any():
                print(f"❌ NaN in movie vector | MovieID={movie_id}")
                return None
            return v
        except Exception as e:
            print(f"❌ Failed to parse vector | MovieID={movie_id} | Error={e}")
            return None

    movies_df["movieVector"] = movies_df.apply(
        lambda r: parse_vector(r["movieVector"], r["ID"]),
        axis=1
    )

    movie_vector_dict = {
        row["ID"]: row["movieVector"]
        for _, row in movies_df.iterrows()
        if row["movieVector"] is not None
    }

    print(f"📦 Loaded {len(movie_vector_dict)} valid movie vectors")

    # ===============================
    # Compute user vectors
    # ===============================
    user_vectors = []

    print("🚀 Calculating user vectors for all users...")
    for uid, user_data in tqdm(df.groupby("UserID"), total=df["UserID"].nunique()):
        weighted_vecs = []
        weights = []

        for _, row in user_data.iterrows():
            movie_vec = movie_vector_dict.get(row["MovieID"])
            if movie_vec is None:
                print(f"⚠ User={uid} | Missing movie vector | MovieID={row['MovieID']}")
                continue

            w = row["Weight"]

            if pd.isna(w):
                print(f"⚠ User={uid} | NaN weight | ActionID={row['ActionID']}")
                continue

            weighted_vecs.append(movie_vec * w)
            weights.append(w)

        if len(weighted_vecs) == 0:
            print(f"⚠ User={uid} | No valid events → skipped")
            continue

        weight_sum = sum(weights)
        if weight_sum == 0:
            print(f"❌ User={uid} | sum(weights)=0 → skipped")
            continue

        user_vec = np.sum(weighted_vecs, axis=0) / weight_sum

        # Detect NaN before normalize
        if np.isnan(user_vec).any():
            print(f"❌ User={uid} | NaN before normalize")
            continue

        # Detect zero vector
        if np.linalg.norm(user_vec) == 0:
            print(f"⚠ User={uid} | Zero vector → skipped")
            continue

        try:
            user_vec = normalize(user_vec.reshape(1, -1))[0]
        except Exception as e:
            print(f"❌ User={uid} | Normalize failed | {e}")
            continue

        user_vectors.append({
            "UserID": uid,
            "userVector": user_vec.tolist()
        })

    # # ===============================
    # # Output
    # # ===============================
    # result = pd.DataFrame(user_vectors)

    # if result.empty:
    #     print("⚠ No user vectors generated → CSV not updated")
    #     return result

    # if save_to_csv and output_path is not None:
    #     output_path.parent.mkdir(parents=True, exist_ok=True)

    #     if output_path.exists():
    #         print(f"ℹ Overwriting existing file: {output_path}")

    #     result.to_csv(output_path, index=False)
    #     print(f"✔ File saved: {output_path}")

    # print(f"✅ Finished | Generated vectors for {len(result)} users")
    # return result
    

    # ===============================
    # Output (SAFE VERSION)
    # ===============================
    result = pd.DataFrame(user_vectors)

    if result.empty:
        print("⚠ No new user vectors → keeping old file")
        return result

    if save_to_csv and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            old_df = pd.read_csv(output_path)
            old_df["UserID"] = old_df["UserID"].astype(int)
            result["UserID"] = result["UserID"].astype(int)

            merged = (
                pd.concat([old_df, result])
                .drop_duplicates(subset="UserID", keep="last")
                .sort_values("UserID")
            )
        else:
            merged = result

        merged.to_csv(output_path, index=False)
        print(f"✔ File updated safely: {output_path}")

    print(f"✅ Finished | Total users in file: {len(merged)}")
    return merged


def run_pipeline():
    print("🔄 Loading events...")
   
    movies_df = load_movie_vectors()
    user_profiles_df = load_user_profiles()
    user_actions_df = load_user_actions()


   
    compute_user_profile_vectors(
        user_profiles_df=user_profiles_df,
        user_actions_df=user_actions_df,
        movies_df=movies_df,
        output_path=OUTPUT_PATH
    )

    mark_events_processed(user_profiles_df["id"].astype(int).tolist())
    print("✅ Events marked as processed")
