import pandas as pd
import os

input_file = r'data/raw/movies.dat'
output_file = r'data/raw/movies.csv'

valid_genres = {
    'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
}

try:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found at {os.path.abspath(input_file)}")
    
    df = pd.read_csv(input_file, sep='::', header=None, names=['ID', 'Title', 'Genres'], encoding='latin1', engine='python')

    duplicates = df[df['ID'].duplicated(keep=False)]
    if not duplicates.empty:
        print(f"Warning: Found {len(duplicates)} duplicate MovieIDs:")
        print(duplicates[['ID', 'Title', 'Genres']])

    def validate_genres(genres):
        if pd.isna(genres):
            return False, "Empty genres"
        genre_list = genres.split('|')
        invalid_genres = [g for g in genre_list if g not in valid_genres]
        if invalid_genres:
            return False, f"Invalid genres: {invalid_genres}"
        return True, None

    df['Genre_Valid'], df['Genre_Error'] = zip(*df['Genres'].apply(validate_genres))
    
    invalid_genres_df = df[~df['Genre_Valid']]
    if not invalid_genres_df.empty:
        print(f"Warning: Found {len(invalid_genres_df)} rows with invalid genres:")
        print(invalid_genres_df[['ID', 'Title', 'Genres', 'Genre_Error']])

    df[['ID', 'Title', 'Genres']].to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Converted {input_file} to {output_file}")
    print(f"Total rows: {len(df)}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")