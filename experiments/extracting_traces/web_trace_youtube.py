import os
import pandas as pd
from datetime import datetime

directory = '/Users/roozbeh/Documents/benchmark_code/files/youtube_traces'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/youtube_resample_1h.csv'

def pars(input):
    normal_dates = []

    # Read the input file and write the modified lines to a new file
    with open(input, 'r') as input_file:
        for line in input_file:
            parts = line.strip().split(' ')
            unix_time = parts[0]
            normal_date = datetime.fromtimestamp(float(unix_time)).strftime('%Y-%m-%d %H:%M:%S')
            normal_dates.append(normal_date)

    # Create a DataFrame from the formatted times
    data = {'Datetime': normal_dates}
    df = pd.DataFrame(data)

    return df


# Get list of files starting with 'youtube.parsed'
files = [file for file in os.listdir(directory) if file.startswith('youtube.parsed')]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate over each file
for file in files:
    file_path = os.path.join(directory, file)
    df = pars(file_path)
    dfs.append(df) 

df = pd.concat(dfs, ignore_index=True)

# Convert 'Datetime' column to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Filter out times before 2007-05-21 based on documentation
df = df[df['Datetime'] >= '2007-05-21']

df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract only the date part
df['Date'] = df['Datetime'].dt.date

# Sort the DataFrame based on the 'Formatted Time' column
df.sort_values(by='Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Set the Datetime column as the index
df.set_index('Datetime', inplace=True)

# Resample the data to get log counts per hour
log_counts_per_hour = df.resample('H').size().dropna()

# Filter out rows where the value is 0
log_counts_per_hour = log_counts_per_hour[log_counts_per_hour != 0]

# Write the resampled data to a CSV file
log_counts_per_hour.to_csv(output_csv_path)