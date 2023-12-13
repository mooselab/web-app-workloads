import re
import pandas as pd
from datetime import datetime

log_file_path = '/Users/roozbeh/Documents/benchmark_code/files/nasa_access_log_2.txt'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/nasa_2_resample_1h.csv'

# Regular expression pattern to match content within square brackets
datetime_pattern = r'\[(.*?)\]'

# List to store extracted date and time values
datetime_values = []

# Custom function to parse the datetime
def custom_datetime_parser(dt_str):
    return datetime.strptime(dt_str, '%d/%b/%Y:%H:%M:%S')

# Open the log file with 'latin-1' encoding
with open(log_file_path, 'r', encoding='latin-1') as file:
    datetime_values = []

    # Iterate through the log lines
    for log_line in file:
        match = re.search(datetime_pattern, log_line)
        if match:
            datetime_part = match.group(1).split()[0]
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