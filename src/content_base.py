import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, csv_path):
        self.movies_df = pd.read_csv(csv_path)
        self.movies_df['Genres'] = self.movies_df['Genres'].fillna('').apply(lambda x: x.split('|'))
        
        self.create_binary_matrix() 
        self.create_tfidf_matrix() 

    def create_binary_matrix(self):
        mlb = MultiLabelBinarizer()
        self.binary_matrix = pd.DataFrame(
            mlb.fit_transform(self.movies_df['Genres']),
            columns=mlb.classes_,
            index=self.movies_df['Title']
        )

        print("\n=== BẢNG THỐNG KÊ 0/1 CỦA THỂ LOẠI ===")
        print(self.binary_matrix.head(8))

    def create_tfidf_matrix(self):
        self.movies_df['Genres_str'] = self.movies_df['Genres'].apply(lambda x: ' '.join(x))
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['Genres_str'])
        self.tfidf_df = pd.DataFrame(
            self.tfidf_matrix.toarray(),
            columns=self.tfidf.get_feature_names_out(),
            index=self.movies_df['Title']
        )

        print("\n=== BẢNG TF-IDF CỦA THỂ LOẠI ===")
        print(self.tfidf_df.head(8))

        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['Title']).drop_duplicates()
        
    def recommend(self, title, num_recommendations=10):
        if title not in self.indices:
            return f"Không tìm thấy phim '{title}' trong dữ liệu."

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]

        return self.movies_df['Title'].iloc[movie_indices].tolist()
