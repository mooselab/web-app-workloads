import pandas as pd
from datetime import datetime

csv_file_path = '/Users/roozbeh/Documents/benchmark_code/files/retailrocket_access_log.csv'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/retailrocket_resample_1h.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Convert hex timestamp to datetime format
df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# Sort the DataFrame by the modified Datetime column
df.sort_values(by='Datetime', inplace=True)

# Set the Datetime column as the index
df.set_index('Datetime', inplace=True)

# Resample the data to get log counts per hour
log_counts_per_hour = df.resample('H').size()

# Write the resampled data to a CSV file
log_counts_per_hour.to_csv(output_csv_path)