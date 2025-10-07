import pandas as pd
import os 

output_file = r'data/raw/movies.csv'

# Bước 1: đọc dữ liệu
movies = pd.read_csv(output_file)

# Bước 2: xem sơ bộ
print(movies.head())
print(movies.info())
print(movies.isna().sum())
