import pandas as pd
from collections import defaultdict

# Read the CSV files into DataFrames
file_path_day = '/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_day.csv'
df_daily = pd.read_csv(file_path_day)
file_path_week = '/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_week.csv'
df_weekly = pd.read_csv(file_path_week)
print(len(df_daily))

# Convert 'Date' column to datetime format in both DataFrames
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_weekly['Date'] = pd.to_datetime(df_weekly['Date'])

# Create a defaultdict to store the counts with the cluster value as the key
weekly_cluster_counts = defaultdict(lambda: defaultdict(int))

# Loop through each row in the weekly dataset
for index, row in df_weekly.iterrows():

    week_start = row['Date']
    week_end = week_start + pd.DateOffset(days=6)  # Calculate the end of the week

    # Filter the daily dataset based on the week's date range
    weekly_rows = df_daily[(df_daily['Date'] >= week_start) & (df_daily['Date'] <= week_end)]

    # Count the occurrences of 'Cluster' in the filtered weekly datasetx
    cluster_counts = weekly_rows['Cluster'].value_counts().to_dict()

    # Sum the counts for each 'Cluster' value for the current week
    for key, value in cluster_counts.items():
        weekly_cluster_counts[row['Cluster']][key] += value

# Display the counts of 'Cluster' in each week with the respective 'Cluster' from the weekly dataset
for cluster, counts in weekly_cluster_counts.items():
    print(f"Cluster {cluster}: {dict(counts)}")