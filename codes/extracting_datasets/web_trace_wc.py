import re
import os
import pandas as pd
from datetime import datetime

folder_path = '/Users/roozbeh/Documents/benchmark_code/files/worldcup_access_log'
output_folder_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/wc98'

# Regular expression pattern to match content within square brackets
datetime_pattern = r'\[(.*?)\]'

# Custom function to parse the datetime
def custom_datetime_parser(dt_str):
    return datetime.strptime(dt_str, '%d/%b/%Y:%H:%M:%S')

# Get a list of log files with '.out' suffix in the folder
log_files = [file for file in os.listdir(folder_path) if file.endswith('.out')]

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Process each log file
for log_file in log_files:
    log_file_path = os.path.join(folder_path, log_file)
    
    # Generate dynamic output path
    log_file_name = os.path.splitext(log_file)[0]
    output_csv_path = os.path.join(output_folder_path, f'{log_file_name}_resample_1h.csv')

    # List to store extracted date and time values
    datetime_values = []

    # Open the log file with 'latin-1' encoding
    with open(log_file_path, 'r', encoding='latin-1') as file:
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

    # Adding 2 hours to the time
    df['Datetime'] = df['Datetime'] + pd.Timedelta(hours=2)

    # Set the Datetime column as the index
    df.set_index('Datetime', inplace=True)

    # Resample the data to get log counts per hour
    log_counts_per_hour = df.resample('H').size()

    # Write the resampled data to a CSV file
    log_counts_per_hour.to_csv(output_csv_path)

    print("Data written to CSV:", output_csv_path)
