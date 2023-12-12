import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax

def perform_clustering(data):
    # Set "Datetime" as the index
    data.set_index('Datetime', inplace=True)

    # Reset the index 
    data.reset_index(inplace=True)

    # Change the format of the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Datetime'])

    # Create a column for the Hour of the dat (0-23)
    data['Hour'] = data['Date'].dt.hour

    # Group by 'File', 'Date', and 'Hour', and sum the 'Value' for each group
    pivot_df = data.groupby(['File', pd.Grouper(key='Date', freq='D'), 'Hour'])['Value'].sum().unstack(fill_value=0).reset_index()

    # Rename the columns to be 1, 2, ..., 24
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    
    # Change columns 1 to 24 to int and replace NaN with 0
    for col in pivot_df.columns[2:]:
        pivot_df[col] = pivot_df[col].fillna(0).astype(int)

    # Get the columns for ml algorithm
    sample_data = pivot_df.iloc[:, 2:].values

    # Calculating skewness and kurtosis for each data point
    skewness_values = []
    kurtosis_values = []

    for i in sample_data:
        skewness_values.append(skew(i))
        kurtosis_values.append(kurtosis(i))

    # Add skewness and kurtosis values to the dataframe
    pivot_df['Skewness'] = skewness_values
    pivot_df['Kurtosis'] = kurtosis_values
    print(pivot_df.head())

    # Normalize the data using Min-Max scaling
    scaler = TimeSeriesScalerMinMax()
    scaled_data = scaler.fit_transform(sample_data)

    # Choose the range of clusters to try
    min_clusters = 2
    max_clusters = 20

    # Initialize variables to store best results
    best_k = min_clusters
    best_silhouette_score = -1
    silhouette_scores = []

    # Loop through different numbers of clusters
    for k in range(min_clusters, max_clusters + 1):
        # Initialize TimeSeriesKMeans model
        model = TimeSeriesKMeans(n_clusters=k, verbose=True, random_state=42)

        # Fit the model to the scaled data
        cluster_labels = model.fit_predict(scaled_data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_data.reshape(len(sample_data), -1), cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For K = {k}, Silhouette Score: {silhouette_avg:.4f}")

        # Check if this K has a better silhouette score
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_k = k

    print(f"Best K: {best_k} with Silhouette Score: {best_silhouette_score:.4f}")

    # Plot the Silhouette Score for different values of K
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title('Silhouette Score vs. Number of Clusters (K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid()

    # Apply TimeSeriesKMeans clustering with the best K
    best_model = TimeSeriesKMeans(n_clusters=best_k, verbose=True, random_state=42)
    best_cluster_labels = best_model.fit_predict(scaled_data)

    # Count the number of samples in each cluster
    unique, counts = np.unique(best_cluster_labels, return_counts=True)
    sample_counts = dict(zip(unique, counts))
    print("Number of samples in each cluster:")
    for cluster_id, count in sample_counts.items():
        print(f"Cluster {cluster_id}: {count} samples")

    # Extract 'Date' and 'File' columns from pivot_df
    pivot_df_subset = pivot_df[['Date', 'File', 'Skewness', 'Kurtosis']].copy()

    # Convert 'Date' column to datetime and add day column
    pivot_df_subset['Date'] = pd.to_datetime(pivot_df_subset['Date'])  
    pivot_df_subset['Day'] = pivot_df_subset['Date'].dt.day_name()

    # Create a DataFrame to associate each sample with its cluster
    pivot_df_subset['Cluster'] = best_cluster_labels

    # Re-order the columns
    pivot_df_subset = pivot_df_subset[['Date', 'Day', 'File', 'Cluster', 'Skewness', 'Kurtosis']]

    # Print the DataFrame
    print(pivot_df_subset.head(10))

    # Save pivot_df_subset to a CSV file
    # pivot_df_subset.to_csv('/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_day.csv', index=False)

    # Adjusting the font and sizes of plt
    plt_settings = {
        'axes.labelsize': 30,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family': 'Times New Roman',
        'font.size': 40,
    }
    
    # Plot real instances
    for cluster_id in range(best_k):
        with plt.rc_context(rc=plt_settings): # Only apply the plt_settings for this plot
            plt.figure(figsize=(8, 6))
            cluster_samples = scaled_data[best_cluster_labels == cluster_id] 
            for sample in cluster_samples:
                plt.plot(sample.ravel(), 'k-', alpha=0.5)
            plt.tight_layout()

            file_name = f'/Users/roozbeh/Documents/benchmark_code/clustered_results/cluster_d{cluster_id + 1}.pdf'
            plt.savefig(file_name, format='pdf', bbox_inches='tight')
            plt.show()

    # Find the centroids
    centroids = best_model.cluster_centers_

    # Combine all centroids into a single DataFrame
    centroid_df = pd.DataFrame({'Cluster_' + str(i + 1): centroid.ravel() for i, centroid in enumerate(centroids)})

    # Save the combined centroids to a CSV file
    centroid_df.to_csv('/Users/roozbeh/Documents/benchmark_code/clustered_results/kmeans_centroids_day.csv', index=False)

    # Create subplots for each centroid with 2 centroids per row
    num_clusters = len(centroids)
    num_rows = num_clusters // 2 + num_clusters % 2  # Calculate the number of rows
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6))

    # Flatten the axs array if it's 2D (only one row)
    if num_rows == 1:
        axs = axs.reshape(1, -1)

    # Plot each centroid as a time series in a subfigure
    for cluster_id, centroid in enumerate(centroids):
        row = cluster_id // 2  # Determine the row
        col = cluster_id % 2   # Determine the column
        axs[row, col].plot(np.arange(24), centroid.ravel())
        axs[row, col].set_title(f'Cluster {cluster_id + 1}')

    # Add labels and a legend
    plt.xlabel('Hour of the Day')
    plt.ylabel('Value')
    plt.suptitle('Centroids of Clusters')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for suptitle

    # Show the plot
    plt.show()

# Define the directory path
directory_path = "/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h"

# Check if the directory exists
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # Use glob to find all CSV files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**', '*.csv'), recursive=True)

    if csv_files:
        # Initialize an empty list to store DataFrames
        df_list = []

        for csv_file in csv_files:
            # Extract the part of the filename before "_resample"
            file_name = csv_file.split("/")[-1].split("_resample")[0].split('_')[0]

            # Read each CSV file into a DataFrame, skipping the header row
            file_path = os.path.join(directory_path, csv_file)
            df = pd.read_csv(file_path, skiprows=[0], names=["Datetime", "Value"])

            # Add a "File" column with the extracted name
            df["File"] = file_name

            # Append the DataFrame to the list
            df_list.append(df)

        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)
        print('length:', len(combined_df))

        # Call the clustering function with the combined DataFrame
        perform_clustering(combined_df)
    else:
        print("No CSV files found in the directory.")
else:
    print(f"The directory {directory_path} does not exist or is not a valid directory.")