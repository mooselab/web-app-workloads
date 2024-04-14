import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.manifold import TSNE
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt_settings = {
    'font.size': 50,
    'xtick.labelsize': 45,
    'ytick.labelsize': 45,
    'font.family': 'Times New Roman'
}

def perform_clustering(data):
    # Set 'Datetime' as the index
    data.set_index('Datetime', inplace=True)

    # Reset the index 
    data.reset_index(inplace=True)

    # Change the format of the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Datetime'])

    # Create a column for the Hour of the day (0-23)
    data['Hour'] = data['Date'].dt.hour

    # Group by 'File', 'Date', and 'Hour', and sum the 'Value' for each group
    pivot_df = data.groupby(['File', pd.Grouper(key='Date', freq='D'), 'Hour'])['Value'].sum().unstack(fill_value=0).reset_index()

    # Rename the columns to be 1, 2, ..., 24
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    print('Dataset length:', len(pivot_df))
    
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

    # Normalize the data using z-score scaling
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data_zscore = scaler.fit_transform(sample_data)
    reshaped_data_zsore = np.squeeze(scaled_data_zscore, axis=-1)
    df_scaled_data_zscore = pd.DataFrame(reshaped_data_zsore)

    # Apply Exponential Moving Average (EMA) smoothing
    scaled_zscore_ema = df_scaled_data_zscore.ewm(span=12, axis=1).mean().values

    # Choose the range of clusters to try
    min_clusters = 3
    max_clusters = 3

    # Initialize variables to store best results
    best_k = min_clusters
    best_silhouette_score = -1
    silhouette_scores = []

    # Loop through different numbers of clusters
    for k in range(min_clusters, max_clusters + 1):
        # Initialize TimeSeriesKMeans model
        model = TimeSeriesKMeans(n_clusters=k, metric='euclidean', verbose=True, random_state=42)

        # Fit the model to the scaled data
        cluster_labels = model.fit_predict(scaled_zscore_ema)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_zscore_ema.reshape(len(scaled_zscore_ema), -1), cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f'For K = {k}, Silhouette Score: {silhouette_avg:.4f}')

        # Check if this K has a better silhouette score
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_k = k

    print(f'Best K: {best_k} with Silhouette Score: {best_silhouette_score:.4f}')

    # Plot the Silhouette Score for different values of K
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title('Silhouette Score vs. Number of Clusters (K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()

    # Apply TimeSeriesKMeans clustering with the best K
    best_model = TimeSeriesKMeans(n_clusters=best_k, metric='euclidean', verbose=True, random_state=42)
    best_cluster_labels = best_model.fit_predict(scaled_zscore_ema)
    
    # Count the number of samples in each cluster
    unique, counts = np.unique(best_cluster_labels, return_counts=True)
    sample_counts = dict(zip(unique, counts))
    print('Number of samples in each cluster:')
    for cluster_id, count in sample_counts.items():
        print(f'Cluster {cluster_id}: {count} samples')
    
    # Initialize a dictionary to store data points for each cluster
    cluster_data_points = {cluster_id: [] for cluster_id in np.unique(best_cluster_labels)}

    # Iterate through each data point and its corresponding cluster label
    for data_point, cluster_label in zip(scaled_zscore_ema, best_cluster_labels):
        # Append the data point to the corresponding cluster in the dictionary
        cluster_data_points[cluster_label].append(data_point)

    # Define cubic function
    def cubic_model(t, a, b, c, d):
        return a * t**3 + b * t**2 + c * t + d
    
    # Calculate the centroids of the clusters
    centroids = best_model.cluster_centers_

    # Convert lists to arrays for each cluster
    for cluster_id, data_points_list in cluster_data_points.items():
        cluster_data_points[cluster_id] = np.array(data_points_list)

        # Create time variable representing hours
        time = np.arange(cluster_data_points[cluster_id].shape[1])

        # Select the centroid of the current cluster
        centroid = centroids[cluster_id].flatten()

        # Fit cubic model
        popt, _ = curve_fit(cubic_model, time, centroid)

        # Extract coefficients
        a, b, c, d = popt

        # Print coefficients of the cubic model
        print('Cubic model coefficients:')
        print('a:', a, ', b:', b, ', c:', c, ', d:', d)

        # Generate data points for cubic model
        cubic_model_data = cubic_model(time, *popt)

        plt.figure(figsize=(8, 4))
        plt.plot(time + 1, centroid, label='Cluster Centroid', linewidth=10, color='cornflowerblue')
        plt.plot(time + 1, cubic_model_data, label='Polynomial Model', linewidth=10, linestyle='--', color='salmon')
        plt.yticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])
        plt.ylim(-1.5, 1.5)  
        plt.fontfamily = 'Times New Roman'
        plt.xticks([0, 6, 12, 18, 24], ['0', '6', '12', '18', '24'], fontsize=30)
        plt.yticks(fontsize=30)
        if cluster_id == 0:
            plt.legend(loc=2, fontsize=30)

        plt.grid(axis='y')
        plt.tight_layout()

        file_name = f'/Users/roozbeh/Documents/benchmark_code/clustered_results/quadratic_cluster_d{cluster_id + 1}.pdf'
        plt.savefig(file_name, format='pdf', bbox_inches='tight')
        plt.show()

    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(scaled_zscore_ema)

    # Plot the t-SNE transformed data, coloring by cluster labels
    plt.figure(figsize=(10, 8))
    for cluster_id in range(best_k):
        cluster_samples = tsne_data[best_cluster_labels == cluster_id]
        plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1], label=f'Cluster {cluster_id}')
    plt.title('t-SNE Visualization of Clustered Time Series Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

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
    pivot_df_subset.to_csv('/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_day.csv', index=False)

    # Plot real instances
    for cluster_id in range(best_k):
        with plt.rc_context(rc=plt_settings):
            plt.figure(figsize=(8, 8))
            cluster_samples = scaled_zscore_ema[best_cluster_labels == cluster_id] 
            for sample in cluster_samples:
                plt.plot(sample.ravel(), 'k-', alpha=0.5)
            plt.ylim(-2, 2)  
            plt.grid(True)
            plt.tight_layout()

            file_name = f'/Users/roozbeh/Documents/benchmark_code/clustered_results/cluster_d{cluster_id + 1}.pdf'
            plt.savefig(file_name, format='pdf', bbox_inches='tight')
            plt.show()
    
    # Reshape the data into a 2D array
    reshaped_data = scaled_zscore_ema.reshape(scaled_zscore_ema.shape[0], -1)

    for cluster_id in range(best_k):
        plt.figure(figsize=(8, 4))
        # Create boxplot
        bp = plt.boxplot(reshaped_data[best_cluster_labels == cluster_id], showfliers=False, widths=0.5,
                        patch_artist=True, boxprops=dict(facecolor='teal', color='teal', linewidth=3),
                        medianprops=dict(color='darkslategrey', linewidth=2),
                        whiskerprops=dict(color='teal', linewidth=2),
                        capprops=dict(color='teal', linewidth=2),
                        meanprops=dict(marker='.', markerfacecolor='red', markeredgecolor='red'))
        
        # Set ticks and show plot
        plt.xticks([0, 6, 12, 18, 24], ['0', '6', '12', '18', '24'])
        plt.grid(axis='y')
        plt.fontfamily = 'Times New Roman'
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        file_name = f'/Users/roozbeh/Documents/benchmark_code/clustered_results/clusterrr_d{cluster_id + 1}.pdf'
        plt.savefig(file_name, format='pdf', bbox_inches='tight')
        plt.show() 

    # Find the centroids
    centroids = best_model.cluster_centers_

    # Combine all centroids into a single DataFrame
    centroid_df = pd.DataFrame({'Cluster_' + str(i + 1): centroid.ravel() for i, centroid in enumerate(centroids)})

    # Save the combined centroids to a CSV file
    centroid_df.to_csv('/Users/roozbeh/Documents/benchmark_code/clustered_results/kmeans_centroids_day.csv', index=False)

# Define the directory path
directory_path = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h'

# Check if the directory exists
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # Use glob to find all CSV files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**', '*.csv'), recursive=True)

    if csv_files:
        # Initialize an empty list to store DataFrames
        df_list = []

        for csv_file in csv_files:
            # Extract the part of the filename before '_resample'
            file_name = csv_file.split('/')[-1].split('_resample')[0].split('_')[0]

            # Read each CSV file into a DataFrame, skipping the header row
            file_path = os.path.join(directory_path, csv_file)
            df = pd.read_csv(file_path, skiprows=[0], names=['Datetime', 'Value'])

            # Add a 'File' column with the extracted name
            df['File'] = file_name

            # Append the DataFrame to the list
            df_list.append(df)

        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)

        # Call the clustering function with the combined DataFrame
        perform_clustering(combined_df)
    else:
        print('No CSV files found in the directory.')
else:
    print(f'The directory {directory_path} does not exist or is not a valid directory.')