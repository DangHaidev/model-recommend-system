import csv

input_file = r'data/raw/ratings.dat'
output_file = r'data/raw/ratings.csv'

try:
    with open(input_file, 'r', encoding='latin1') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.writer(outfile)
        # Chỉ giữ 4 cột gốc
        writer.writerow(['UserID', 'MovieID', 'Rating', 'Timestamp'])
        
        for line in infile:
            try:
                user_id, movie_id, rating, timestamp = line.strip().split('::')
                writer.writerow([user_id, movie_id, rating, timestamp])
            except ValueError as e:
                print(f"Skipping line due to error: {line.strip()} - {e}")
            except Exception as e:
                print(f"Error processing line {line.strip()}: {e}")

    print(f"✅ Converted {input_file} to {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
