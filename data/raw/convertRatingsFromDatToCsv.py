import csv
from datetime import datetime
import pytz

input_file = r'data/raw/ratings.dat'
output_file = r'data/raw/ratings.csv'

try:
    with open(input_file, 'r', encoding='latin1') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['UserID', 'MovieID', 'Rating', 'Formatted_Timestamp'])
        for line in infile:
            try:
                user_id, movie_id, rating, timestamp = line.strip().split('::')
                utc_dt = datetime.fromtimestamp(int(timestamp), tz=pytz.UTC)
                local_tz = pytz.timezone('Asia/Ho_Chi_Minh')
                local_dt = utc_dt.astimezone(local_tz)
                formatted_time = local_dt.strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([user_id, movie_id, rating, formatted_time])
            except ValueError as e:
                print(f"Skipping line due to error: {line.strip()} - {e}")
            except Exception as e:
                print(f"Error processing line {line.strip()}: {e}")
except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")