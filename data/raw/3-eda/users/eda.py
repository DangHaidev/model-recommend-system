import pandas as pd
import os 

input_file = r'data/raw/users.csv'

# Bước 1: đọc dữ liệu
users = pd.read_csv(input_file)

print(users.info())
print(users.isna().sum())
print(users.duplicated().sum())
