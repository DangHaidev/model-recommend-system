import pandas as pd
import os 

input_file = r'data/raw/movies.csv'

# Bước 1: đọc dữ liệu
movies = pd.read_csv(input_file)

print(movies.info())
print(movies.isna().sum())
print(movies.duplicated().sum())
