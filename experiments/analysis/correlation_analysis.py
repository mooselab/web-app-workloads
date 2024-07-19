import pandas as pd
from collections import defaultdict

# Read the CSV files into DataFrames
file_path_day = '/results/clustered_kmeans_day.csv'
df_daily = pd.read_csv(file_path_day)
file_path_week = '/results/clustered_kmeans_week.csv'
df_weekly = pd.read_csv(file_path_week)

# Convert 'Date' column to datetime format in both DataFrames
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_weekly['Date'] = pd.to_datetime(df_weekly['Date'])

# Create a defaultdict to store the counts with the cluster value as the key
weekly_cluster_counts = defaultdict(lambda: defaultdict(int))
weekly_files_counts = defaultdict(lambda: defaultdict(set))

# Loop through each row in the weekly dataset
for index, row in df_weekly.iterrows():

    # Calculate the start and end of the week
    week_start = row['Date']
    week_end = week_start + pd.DateOffset(days=6)  

    # Filter the daily dataset based on the week's date range
    weekly_rows = df_daily[(df_daily['Date'] >= week_start) & (df_daily['Date'] <= week_end)]

    # Count the occurrences of 'Cluster' in the filtered weekly datasetx
    cluster_counts = weekly_rows['Cluster'].value_counts().to_dict()

    # Sum the counts for each 'Cluster' value for the current week
    for key, value in cluster_counts.items():
        weekly_cluster_counts[f'w{row["Cluster"] + 1}'][f'd{key + 1}'] += value

    # Get unique files for each daily pattern within the week
    unique_files = weekly_rows.groupby('Cluster')['File'].unique()

    # Update the defaultdict with the unique files for each daily cluster within the week
    for daily_cluster, files in unique_files.items():
        weekly_files_counts[f'w{row["Cluster"] + 1}'][f'd{daily_cluster + 1}'].update(files)

# Convert the dictionary into a DataFrame and transpose it
df = pd.DataFrame(weekly_cluster_counts).fillna(0)
df = df.T

# Calculate the total count
total_count = df.values.sum()

# Convert raw counts to percentages
df_percentage = (df / total_count) * 100

# # Sort rows and columns
df_percentage = df_percentage.reindex(sorted(df_percentage.index, key=lambda x: int(x[1:]))[::-1], axis=0)
df_percentage = df_percentage.reindex(sorted(df_percentage.columns, key=lambda x: int(x[1:])), axis=1)
print('Relative frequency of daily and weekly clusters:')
print(df_percentage)

# Convert the defaultdict into a DataFrame and transpose it
df_files = pd.DataFrame(weekly_files_counts)

# Count the number of unique files for each daily and weekly pattern
df_files_count = df_files.applymap(lambda x: len(x))

# Print the results for files count
for week_cluster, daily_files in weekly_files_counts.items():
    print(f"Weekly Cluster: {week_cluster}")
    for daily_cluster, files in daily_files.items():
        print(f"\tDaily Cluster: {daily_cluster}, Unique Files: {files}")

print('Frequency of unique datasets for daily and weekly clusters:')
print(df_files_count)