from src.content_base import ContentBasedRecommender

csv_path = "data/processed/movies_after_preprocessing.csv"

recommender = ContentBasedRecommender(csv_path)

movie_name = "Toy Story (1995)"
recommendations = recommender.recommend(movie_name, 10)

print(f"\nPhim tương tự với '{movie_name}':")
for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")
