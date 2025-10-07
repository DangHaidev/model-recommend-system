import pandas as pd
import os 

input_file = r'data/raw/ratings.csv'

# Bước 1: đọc dữ liệu
ratings = pd.read_csv(input_file)

print(ratings.info())
print(ratings.isna().sum())
print(ratings.duplicated().sum())
