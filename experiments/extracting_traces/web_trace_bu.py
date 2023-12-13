import os
import pandas as pd
from datetime import datetime

# Specify the path to the parent directory containing log files and subdirectories
parent_directory = '/Users/roozbeh/Downloads/condensed/'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/bu_resample_1h.csv'

# Initialize an empty list to store extracted times
extracted_times = []

# Traverse through all subdirectories and files
for root, dirs, files in os.walk(parent_directory):
    for filename in files:
        if filename != '.DS_Store':
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='latin-1') as file:
                for line in file:
                    parts = line.split()
                    extracted_times.append(parts[1])

# Convert extracted times to datetime format
formatted_times = [datetime.utcfromtimestamp(int(time)).strftime('%Y-%m-%d %H:%M:%S') for time in extracted_times]

# Create a DataFrame from the formatted times
data = {'Datetime': formatted_times}
df = pd.DataFrame(data)

# Convert 'Datetime' column to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Filter out times before 1994-11-20 based on documentation
df = df[df['Datetime'] >= '1994-11-20']

# Sort the DataFrame based on the 'Formatted Time' column
df.sort_values(by='Datetime', inplace=True)

# Set the Datetime column as the index
df.set_index('Datetime', inplace=True)

# Resample the data to get log counts per hour
log_counts_per_hour = df.resample('H').size()

# Write the resampled data to a CSV file
log_counts_per_hour.to_csv(output_csv_path)