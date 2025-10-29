import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class HybridRecommender:
    def __init__(self, content_model, cf_model, alpha=0.5):
        """
        content_model: instance của ContentBasedRecommender
        cf_model: instance của CF
        alpha: trọng số giữa content và collaborative (0.0 → chỉ CF, 1.0 → chỉ Content)
        """
        self.content_model = content_model
        self.cf_model = cf_model
        self.alpha = alpha
        self.scaler = MinMaxScaler()

    def recommend(self, user_id, liked_title, top_n=10, verbose=True):
        """
        Gợi ý top_n phim cho user_id dựa trên phim liked_title
        """
        if liked_title not in self.content_model.indices:
            return f"Không tìm thấy phim '{liked_title}' trong dữ liệu content."

        movie_idx = self.content_model.indices[liked_title]

        sim_scores = list(enumerate(self.content_model.cosine_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:]]  

        cf_scores = []
        for i in movie_indices:
            try:
                score = self.cf_model.pred(user_id, i)
                cf_scores.append(score)
            except:
                cf_scores.append(0)

        sim_values = np.array([x[1] for x in sim_scores[1:len(cf_scores)+1]]).reshape(-1, 1)
        cf_values = np.array(cf_scores).reshape(-1, 1)

        sim_scaled = self.scaler.fit_transform(sim_values)
        cf_scaled = self.scaler.fit_transform(cf_values)

        hybrid_score = self.alpha * sim_scaled + (1 - self.alpha) * cf_scaled

        movie_candidates = self.content_model.movies_df.iloc[movie_indices].copy()
        movie_candidates['ContentScore'] = sim_scaled
        movie_candidates['CFScore'] = cf_scaled
        movie_candidates['HybridScore'] = hybrid_score
        movie_candidates = movie_candidates.sort_values('HybridScore', ascending=False)

        if verbose:
            print(f"\n=== Điểm chi tiết cho user {user_id} dựa trên '{liked_title}' ===")
            for idx, row in movie_candidates.head(top_n).iterrows():
                print(f"{row['Title']:<40} | Content: {row['ContentScore']:.4f} | CF: {row['CFScore']:.4f} | Hybrid: {row['HybridScore']:.4f}")

        return movie_candidates[['Title', 'Genres', 'ContentScore', 'CFScore', 'HybridScore']].head(top_n)
