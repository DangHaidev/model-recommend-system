import pandas as pd
import os

input_file = r'data/raw/users.dat'
output_file = r'data/raw/users.csv'

age_mapping = {
    1: 'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-44',
    45: '45-49',
    50: '50-55',
    56: '56+'
}

occupation_mapping = {
    0: 'other or not specified',
    1: 'academic/educator',
    2: 'artist',
    3: 'clerical/admin',
    4: 'college/grad student',
    5: 'customer service',
    6: 'doctor/health care',
    7: 'executive/managerial',
    8: 'farmer',
    9: 'homemaker',
    10: 'K-12 student',
    11: 'lawyer',
    12: 'programmer',
    13: 'retired',
    14: 'sales/marketing',
    15: 'scientist',
    16: 'self-employed',
    17: 'technician/engineer',
    18: 'tradesman/craftsman',
    19: 'unemployed',
    20: 'writer'
}

gender_mapping = {
    'M': 0,
    'F': 1
}

if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found. Please check the path or file name.")
else:
    df = pd.read_csv(
        input_file,
        sep='::',
        header=None,
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
        encoding='latin1',
        engine='python'   # bắt buộc nếu sep='::'
    )

    # Mapping dữ liệu
    df['Gender'] = df['Gender'].map(gender_mapping)
    df['Age'] = df['Age'].map(age_mapping)
    df['Occupation'] = df['Occupation'].map(occupation_mapping)

    # Xuất CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Converted {input_file} to {output_file}")
    print(f"Total rows: {len(df)}")
