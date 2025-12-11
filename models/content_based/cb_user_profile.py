import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_loader import (
    load_movie_vectors,
    load_user_profiles,
    load_user_actions,
    load_user_profile_vectors
)

RELEVANT_THRESHOLD = 2.0

movies = load_movie_vectors() 
user_vectors_df = load_user_profile_vectors()  
user_profile = load_user_profiles()
user_action = load_user_actions()

movies['movieVector'] = movies['movieVector'].apply(lambda x: np.array(eval(x)))
user_vectors_df['userVector'] = user_vectors_df['userVector'].apply(lambda x: np.array(eval(x)))

df = user_profile.merge(user_action, left_on='ActionID', right_on='ID', how='left')

def relevant_movies(user_id):
    data = df[df['UserID']==user_id]
    rel = data[data['Weight']>=RELEVANT_THRESHOLD]['MovieID'].unique()
    return set(rel)

def apply_ratio(movie_vecs, user_vec, wm, wu):
    return movie_vecs * wm, user_vec * wu

def recommend_movies(user_id, wm=1.0, wu=1.0, top_n=10):
    uvec = user_vectors_df.loc[user_vectors_df['UserID']==user_id,'userVector'].values[0]
    mvecs = np.stack(movies['movieVector'].values)

    mvecs, uvec = apply_ratio(mvecs, uvec, wm, wu)
    uvec = uvec.reshape(1, -1)

    sims = -np.sum(np.abs(mvecs - uvec), axis=1)  
    movies['sim'] = sims

    return movies.sort_values('sim', ascending=False).head(top_n)[['ID','Title','sim']]

if __name__ == "__main__":
    user_id = 1
    top_movies = recommend_movies(user_id, wm=1.0, wu=1.0, top_n=10)
    print(f"Top {10} recommended movies for user {user_id} using Manhattan distance:")
    print(top_movies)
