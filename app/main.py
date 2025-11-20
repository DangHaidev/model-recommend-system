from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ============================
# Load dữ liệu
# ============================
print("Loading CSV data...")

movies = pd.read_csv("vector_model/movie_vector.csv")
user_vectors_df = pd.read_csv("vector_model/user_profile_vector.csv")
user_profile = pd.read_csv("vector_model/user_profiles.csv")
user_actions = pd.read_csv("vector_model/user_actions.csv")

# Convert string-list → numpy array
movies["movieVector"] = movies["movieVector"].apply(lambda x: np.array(eval(x)))
user_vectors_df["userVector"] = user_vectors_df["userVector"].apply(lambda x: np.array(eval(x)))

# Merge để xác định relevant movie
df = user_profile.merge(user_actions, left_on="ActionID", right_on="ID", how="left")

RELEVANT_THRESHOLD = 3.0


def relevant_movies(user_id):
    data = df[df["UserID"] == user_id]
    rel = data[data["Weight"] >= RELEVANT_THRESHOLD]["MovieID"].unique()
    return set(rel)


# ============================
# Utility
# ============================
def apply_ratio(movie_vecs, user_vec, wm, wu):
    return movie_vecs * wm, user_vec * wu


# ============================
# Recommend function
# ============================
def recommend_movies(user_id, wm, wu, method, top_n):
    # Lấy userVector
    uvec = user_vectors_df.loc[user_vectors_df["UserID"] == user_id, "userVector"].values[0]

    # Lấy tất cả movie vectors
    mvecs = np.stack(movies["movieVector"].values)

    # Áp dụng trọng số cho user & movie
    mvecs, uvec = apply_ratio(mvecs, uvec, wm, wu)
    uvec = uvec.reshape(1, -1)

    # --- similarity method ---
    if method == "cosine":
        sims = cosine_similarity(uvec, mvecs)[0]

    elif method == "pearson":
        sims = np.array([
            np.corrcoef(uvec.flatten(), mv.flatten())[0, 1]
            for mv in mvecs
        ])
        sims = np.nan_to_num(sims)

    elif method == "jaccard":
        ub = (uvec > 0).astype(int)
        mb = (mvecs > 0).astype(int)
        sims = np.array([
            np.sum(np.minimum(ub, mv)) / np.sum(np.maximum(ub, mv))
            for mv in mb
        ])

    elif method == "manhattan":
        sims = -np.sum(np.abs(mvecs - uvec), axis=1)

    else:
        raise ValueError("Invalid method")

    movies["sim"] = sims

    result = movies.sort_values("sim", ascending=False)[["ID", "Title", "sim"]].head(top_n)

    return result.to_dict(orient="records")


# ============================
# FastAPI
# ============================

app = FastAPI()


class RecommendRequest(BaseModel):
    user_id: int
    wm: float = 1.0
    wu: float = 1.0
    method: str = "cosine"
    top_n: int = 10


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        result = recommend_movies(
            user_id=req.user_id,
            wm=req.wm,
            wu=req.wu,
            method=req.method,
            top_n=req.top_n
        )
        return {"user_id": req.user_id, "results": result}
    except Exception as e:
        return {"error": str(e)}
