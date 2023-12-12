import pandas as pd
import re

log_file_path = '/Users/roozbeh/Documents/benchmark_code/files/sdsc_access_log.txt'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/sdsc_resample_1h.csv'

# Initialize a list to store extracted date-time strings
datetime_strings = []

# Define a regex pattern to match the date-time format
date_time_pattern = r'([A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4})'

# Extract date-time strings using regex
with open(log_file_path, 'r', encoding='latin-1') as file:
    for line in file:
        match = re.search(date_time_pattern, line)
        if match:
            datetime_string = match.group(1)
            datetime_strings.append(datetime_string)

# Convert date-time strings to the desired format
date_format = '%b %d %H:%M:%S %Y'
formatted_dates = [pd.to_datetime(date, format=date_format) for date in datetime_strings]

# Create a Pandas DataFrame
data = {'Datetime': formatted_dates}
df = pd.DataFrame(data)

# Set the Datetime column as the index
df.set_index('Datetime', inplace=True)

# Resample the data to get log counts per hour
log_counts_per_hour = df.resample('H').size()

# Export the resampled data to a CSV file
log_counts_per_hour.to_csv(output_csv_path)