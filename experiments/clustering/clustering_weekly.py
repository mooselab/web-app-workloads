import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Function to perform clustering and analysis
def perform_clustering(data):
    # Preprocess data
    pivot_df = preprocess_data(data)

    # Normalize and smooth data
    scaled_zscore_ema = normalize_and_smooth(pivot_df)

    # Find optimal number of clusters
    best_k, best_silhouette_score = find_optimal_clusters(scaled_zscore_ema)

    # Cluster data with optimal K
    best_model, best_cluster_labels = cluster_data(scaled_zscore_ema, best_k)

    # Visualize centroids
    centroids = visualize_centroids_helper(best_model, best_cluster_labels, scaled_zscore_ema)

    # Visualize clusters
    visualize_clusters(best_k, best_cluster_labels, scaled_zscore_ema)

    # Save results
    save_results(pivot_df, best_cluster_labels, centroids)

# Function to preprocess data
def preprocess_data(data):
    data.set_index('Datetime', inplace=True)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Datetime'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    pivot_df = data.groupby(['File', pd.Grouper(key='Date', freq='W'), 'DayOfWeek'])['Value'].sum().unstack(fill_value=0).reset_index()
    pivot_df.columns = ['File', 'Date', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_df['Month'] = pivot_df['Date'].astype(str).str.split('-').str[1]
    pivot_df = pivot_df[['File', 'Date', 'Month', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
    for col in pivot_df.columns[2:]:
        pivot_df[col] = pivot_df[col].fillna(0).astype(int)
    pivot_df.to_csv('/Users/roozbeh/Documents/benchmark_code/files/merged_dataset_weekly.csv', index=False)
    return pivot_df

# Function to normalize and smooth data
def normalize_and_smooth(pivot_df):
    sample_data = pivot_df.iloc[:, 3:].values
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data_zscore = scaler.fit_transform(sample_data)
    reshaped_data_zscore = np.squeeze(scaled_data_zscore, axis=-1)
    df_scaled_data_zscore = pd.DataFrame(reshaped_data_zscore)
    scaled_zscore_ema = df_scaled_data_zscore.ewm(span=7, axis=1).mean().values
    return scaled_zscore_ema

# Function to find optimal number of clusters
def find_optimal_clusters(scaled_zscore_ema):
    min_clusters = 3
    max_clusters = 3
    best_k = min_clusters
    best_silhouette_score = -1
    silhouette_scores = []
    for k in range(min_clusters, max_clusters + 1):
        model = TimeSeriesKMeans(n_clusters=k, metric='euclidean', verbose=True, random_state=42)
        cluster_labels = model.fit_predict(scaled_zscore_ema)
        silhouette_avg = silhouette_score(scaled_zscore_ema.reshape(len(scaled_zscore_ema), -1), cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f'For K = {k}, Silhouette Score: {silhouette_avg:.4f}')
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_k = k
    print(f'Best K: {best_k} with Silhouette Score: {best_silhouette_score:.4f}')
    return best_k, best_silhouette_score

# Function to cluster data
def cluster_data(scaled_zscore_ema, best_k):
    best_model = TimeSeriesKMeans(n_clusters=best_k, metric='euclidean', verbose=True, random_state=42)
    best_cluster_labels = best_model.fit_predict(scaled_zscore_ema)
    unique, counts = np.unique(best_cluster_labels, return_counts=True)
    sample_counts = dict(zip(unique, counts))
    print('Number of samples in each cluster:')
    for cluster_id, count in sample_counts.items():
        print(f'Cluster {cluster_id}: {count} samples')
    return best_model, best_cluster_labels

# Define quadratic function
def quadratic_model(t, a, b, c):
    return a * t**2 + b * t + c 

# Function to visualize clusters
def visualize_centroids_helper(best_model, best_cluster_labels, scaled_zscore_ema):
    centroids = best_model.cluster_centers_
    for cluster_id in range(best_model.n_clusters):
        cluster_data_points = scaled_zscore_ema[best_cluster_labels == cluster_id]
        visualize_centroids(cluster_id, centroids[cluster_id], cluster_data_points)
    return centroids

# Function to plot centroids
def visualize_centroids(cluster_id, centroid, cluster_data_points):
    time = np.arange(cluster_data_points.shape[1])
    centroid = centroid.flatten()
    popt, _ = curve_fit(quadratic_model, time, centroid)
    a, b, c = popt
    quadratic_model_data = quadratic_model(time, *popt)
    plt.figure(figsize=(8, 4))
    plt.plot(time + 1, centroid, label='Cluster Centroid', linewidth=10, color='cornflowerblue')
    plt.plot(time + 1, quadratic_model_data, label='Polynomial Model', linewidth=10, linestyle='--', color='salmon')
    plt.yticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])
    plt.ylim(-1.5, 1.5)  
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if cluster_id == 0:
        plt.legend(loc=2, fontsize=30)
    plt.grid(axis='y')
    plt.tight_layout()
    file_name = os.path.join(output_path, f'quadratic_cluster_w{cluster_id + 1}.pdf')
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    plt.show()

def visualize_clusters(best_k, best_cluster_labels, scaled_zscore_ema):
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
        plt.grid(axis='y')
        plt.fontfamily = 'Times New Roman'
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        file_name = os.path.join(output_path, f'clusterrr_w{cluster_id + 1}.pdf')
        plt.savefig(file_name, format='pdf', bbox_inches='tight')
        plt.show() 

# Function to save clustering results
def save_results(pivot_df, best_cluster_labels, centroids):
    pivot_df_subset = pivot_df[['Date', 'File']].copy()
    pivot_df_subset['Date'] = pd.to_datetime(pivot_df_subset['Date'])  
    pivot_df_subset['Day'] = pivot_df_subset['Date'].dt.day_name()
    pivot_df_subset['Cluster'] = best_cluster_labels
    pivot_df_subset = pivot_df_subset[['Date', 'Day', 'File', 'Cluster']]
    pivot_df_subset.to_csv(os.path.join(output_path, 'clustered_kmeans_week.csv'), index=False)

    centroid_df = pd.DataFrame({'Cluster_' + str(i + 1): centroid.ravel() for i, centroid in enumerate(centroids)})
    centroid_df.to_csv(os.path.join(output_path, 'kmeans_centroids_week.csv'), index=False)

### MAIN - Merge all CSV files in the directory
# Define the input path
input_path = '/data'
# Define the base directory for saving files
output_path = '/results'

if os.path.exists(input_path) and os.path.isdir(input_path):
    csv_files = glob.glob(os.path.join(input_path, '**', '*.csv'), recursive=True)
    if csv_files:
        df_list = []
        for csv_file in csv_files:
            file_name = csv_file.split('/')[-1].split('_resample')[0].split('_')[0]
            file_path = os.path.join(input_path, csv_file)
            df = pd.read_csv(file_path, skiprows=[0], names=['Datetime', 'Value'])
            df['File'] = file_name
            df_list.append(df)
        combined_df = pd.concat(df_list, ignore_index=True)
        value_counts = combined_df['File'].value_counts()
        rows_to_remove = value_counts[value_counts < 24*7].index
        filtered_df = combined_df[~combined_df.File.isin(rows_to_remove)]
        sorted_df = filtered_df.sort_values(by=['File', 'Datetime']).reset_index(drop=True)
        perform_clustering(sorted_df)
    else:
        print('No CSV files found in the directory.')
else:
    print(f'The directory {input_path} does not exist or is not a valid directory.')