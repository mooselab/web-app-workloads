import pandas as pd
import matplotlib.pyplot as plt

def clustering_analysis(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Group by 'File' and 'Cluster' and calculate the percentage
    file_cluster_distribution = df.groupby(['File', 'Cluster']).size().unstack(fill_value=0).div(df['File'].value_counts(), axis=0).fillna(0) * 100

    # Group by 'Cluster' and 'File' and calculate the percentage
    cluster_file_distribution = df.groupby(['Cluster', 'File']).size().unstack(fill_value=0).div(df['Cluster'].value_counts(), axis=0).fillna(0) * 100

    if 'clustered_kmeans_day' in file_path:
        # Group by 'Week' and 'Cluster' and calculate the percentage
        time_cluster_distribution = df.groupby(['Day', 'Cluster']).size().unstack(fill_value=0).div(df['Day'].value_counts(), axis=0).fillna(0) * 100

        # Group by 'Cluster' and 'Week' and calculate the percentage
        cluster_time_distribution = df.groupby(['Cluster', 'Day']).size().unstack(fill_value=0).div(df['Cluster'].value_counts(), axis=0).fillna(0) * 100

    elif 'clustered_kmeans_week' in file_path:
        # Group by 'Month' and 'Cluster' and calculate the percentage
        time_cluster_distribution = df.groupby(['Month', 'Cluster']).size().unstack(fill_value=0).div(df['Month'].value_counts(), axis=0).fillna(0) * 100

        # Group by 'Cluster' and 'Month' and calculate the percentage
        cluster_time_distribution = df.groupby(['Cluster', 'Month']).size().unstack(fill_value=0).div(df['Cluster'].value_counts(), axis=0).fillna(0) * 100

    elif 'clustered_kmeans_month' in file_path:
        # Group by 'Year' and 'Cluster' and calculate the percentage
        time_cluster_distribution = df.groupby(['Year', 'Cluster']).size().unstack(fill_value=0).div(df['Year'].value_counts(), axis=0).fillna(0) * 100

        # Group by 'Cluster' and 'Year' and calculate the percentage
        cluster_time_distribution = df.groupby(['Cluster', 'Year']).size().unstack(fill_value=0).div(df['Cluster'].value_counts(), axis=0).fillna(0) * 100

    # Print the results
    print(time_cluster_distribution)
    print(cluster_time_distribution)
    print(file_cluster_distribution)
    print(cluster_file_distribution)

# Choose the daily or weekly file
clustering_analysis('/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_day.csv')
clustering_analysis('/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_week.csv')