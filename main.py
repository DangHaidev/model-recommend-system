import pandas as pd
import numpy as np

from src.content_base import ContentBasedRecommender
from src.collaborative import CF, get_dataframe_ratings_base
from src.hybrid import HybridRecommender

csv_path = "data/processed/movies_after_preprocessing.csv"

print("Khởi tạo mô hình Content-Based Recommender...")
content_model = ContentBasedRecommender(csv_path)

print("\nKhởi tạo mô hình Collaborative Filtering...")
Y_data = get_dataframe_ratings_base()              
cf_model = CF(data_matrix=Y_data, k=5, uuCF=1)     
cf_model.normalize_matrix()
cf_model.similarity()

print("\nKết hợp mô hình Hybrid...")
hybrid = HybridRecommender(content_model, cf_model, alpha=0.6)

user_id = 1
liked_title = "Toy Story (1995)"
top_n = 10

print(f"\nGỢI Ý HYBRID CHO USER {user_id} DỰA TRÊN PHIM '{liked_title}':\n")
recommendations = hybrid.recommend(user_id=user_id, liked_title=liked_title, top_n=top_n)



