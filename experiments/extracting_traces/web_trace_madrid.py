import pandas as pd

log_file_path = '/Users/roozbeh/Documents/benchmark_code/files/madrid_access_log.txt'
output_csv_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/madrid_resample_1h.csv'

import pandas as pd

# Read the text file
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Parse each line and create a list of tuples (datetime, value)
data = [(pd.to_datetime(line.split(',')[0].strip('"')), int(line.split(',')[1])) for line in lines]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Datetime', 'Value'])

# Write the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)