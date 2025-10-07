import pandas as pd
import os 

output_file = r'data/raw/users.csv'

# Bước 1: đọc dữ liệu
users = pd.read_csv(output_file)

# Bước 2: xem sơ bộ
print(users.head())
print(users.info())
print(users.isna().sum())
