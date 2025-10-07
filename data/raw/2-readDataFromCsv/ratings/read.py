import pandas as pd
import os 

output_file = r'data/raw/ratings.csv'

# Bước 1: đọc dữ liệu
ratings = pd.read_csv(output_file)

# Bước 2: xem sơ bộ
print(ratings.head())
print(ratings.info())
print(ratings.isna().sum())
