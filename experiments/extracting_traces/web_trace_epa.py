import re
import pandas as pd
from datetime import datetime

log_file_path = '/Users/roozbeh/Documents/benchmark_code/files/epa/epa_access_log.txt'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/epa/epa_resample_1h.csv'

# Regular expression pattern to match content within brackets
datetime_pattern = r'\[(.*?)\]'

# Custom function to parse the datetime
def custom_datetime_parser(dt_str):
    date_part = '1995-08-' + dt_str[:2]  # Assuming the log lines are from August 1995
    time_part = dt_str[3:]
    return datetime.strptime(f'{date_part} {time_part}', '%Y-%m-%d %H:%M:%S')

# Open the log file with 'latin-1' encoding
with open(log_file_path, 'r', encoding='latin-1') as file:
    datetime_values = []

    # Iterate through the log lines
    for log_line in file:
        match = re.search(datetime_pattern, log_line)
        if match:
            datetime_part = match.group(1)
            datetime_values.append(custom_datetime_parser(datetime_part))
        else:
            datetime_values.append(None)

# Create a Pandas DataFrame
data = {'Datetime': datetime_values}
df = pd.DataFrame(data)

# Set the Datetime column as the index
df.set_index('Datetime', inplace=True)

# Resample the data to get log counts per hour
log_counts_per_hour = df.resample('H').size()

# Write the resampled data to a CSV file
log_counts_per_hour.to_csv(output_csv_path)