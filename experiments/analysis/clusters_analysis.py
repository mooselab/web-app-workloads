import pandas as pd
import matplotlib.pyplot as plt

# Read the DataFrame from the CSV file
centroids = pd.read_csv('/Users/roozbeh/Documents/benchmark_code/clustered_results/kmeans_centroids_week.csv')

# Create an empty DataFrame to store the modified data
centroids_shifted = pd.DataFrame()

# Iterate through each column in the original DataFrame
for column in centroids.columns:
    # Find the index of the minimum value in the column
    min_index = centroids[column].idxmin()
    
    # Split the column into two parts: before and after the minimum value
    before_min = centroids[column].iloc[:min_index]
    after_min = centroids[column].iloc[min_index:]
    
    # Concatenate the two parts, with the minimum value at the beginning
    new_column = pd.concat([after_min, before_min], ignore_index=True)
    
    # Add the modified column to the new DataFrame
    centroids_shifted[column] = new_column

# Print the modified DataFrame
print(centroids_shifted)

# Save the shifted DataFrame to a CSV file
centroids_shifted.to_csv('/Users/roozbeh/Documents/benchmark_code/clustered_results/kmeans_centroids_week_shifted.csv', index=False)

# Create a figure with two subplots in the same row
fig, axs = plt.subplots(1, 2, figsize=(15, 4))

# Plot each column in the first subplot
for column in centroids.columns:
    axs[0].plot(centroids.index, centroids[column], label=f'Column {column}')

# Plot each column in the second subplot (same as the first)
for column in centroids_shifted.columns:
    axs[1].plot(centroids_shifted.index, centroids_shifted[column], label=f'Column {column}')

# Add labels and legend to the first subplot
axs[0].set_title('Original Centroids')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')
axs[0].legend(loc='upper left')

# Add labels and legend to the second subplot (same as the first)
axs[1].set_title('Shifted Centroids')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
axs[1].legend(loc='upper left')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Calculate mean, standard deviation, and CV for each column
column_means = centroids_shifted.mean()
column_medians = centroids_shifted.median()
column_deviations = centroids_shifted.std()
cv = column_deviations / column_means

# Print the results
print("Mean of each column:")
print(column_means)
print("Median of each column:")
print(column_medians)
print("Std of each column:")
print(column_deviations)
print("CV of each column:")
print(cv)

# Calculate the 25th and 75th percentiles for each column
percentiles_25 = centroids_shifted.quantile(0.25)
percentiles_75 = centroids_shifted.quantile(0.75)

# Display the 25th and 75th percentiles for each column
for col in centroids_shifted.columns:
    print(f'Column {col}: 25th percentile = {percentiles_25[col]}, 75th percentile = {percentiles_75[col]}')

# Calculate how many values in each column are between the 0.25 and 0.75 quantiles
between_25_75_counts = ((centroids_shifted >= percentiles_25) & (centroids_shifted <= percentiles_75)).mean()

# Display the percentage of 25-75 quantiles
for col, count in between_25_75_counts.items():
    print(f'Column {col}: Number of values between 25th and 75th percentiles = {count}')

# Calculate how many values in each column are between 0.25 and 0.75
between_025_075_counts = ((centroids_shifted >= 0.25) & (centroids_shifted <= 0.75)).mean()

# Display the percentage of values between 0.25 and 0.75
for col, count in between_025_075_counts.items():
    print(f'Column {col}: Number of values between 0.25 and 0.75 = {count}')

# Calculate the percentage of non-zero values for each column
non_zero_percentage = (centroids_shifted[centroids_shifted != 0].count() / centroids_shifted.count()) * 100

# Display the percentage of non-zero values for each column
for col, percentage in non_zero_percentage.items():
    print(f'Column {col}: Percentage of non-zero values = {percentage:.2f}%')

# Loop through each column in centroids_shifted to find the steep of q1 to q3
for column in centroids_shifted.columns:
    data = centroids_shifted[column]  # Get the column data

    # Calculate the 25th and 75th percentiles (q1 and q3)
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    # Find the indices where the data crosses from q1 to q3
    q1_to_q3_indices = (data >= q1) & (data <= q3)

    # Initialize variables to keep track of continuous sequences
    continuous_true_count = 0
    max_continuous_true_count = 0
    continuous_true_sequences = []

    for value in q1_to_q3_indices:
        if value:
            continuous_true_count += 1
        else:
            if continuous_true_count > 0:
                continuous_true_sequences.append(continuous_true_count)
                max_continuous_true_count = max(max_continuous_true_count, continuous_true_count)
                continuous_true_count = 0

    # In case the last sequence extends to the end of the Series
    if continuous_true_count > 0:
        continuous_true_sequences.append(continuous_true_count)
        max_continuous_true_count = max(max_continuous_true_count, continuous_true_count)

    print('Time to get from q1 to q3 in', column, 'is:', continuous_true_sequences)

def duration_above(data, threshold):
    # Loop through each column in centroids_shifted to find the higher than median values
    for column in data.columns:

        # Calculate the median value
        median_value = data[column].median()
        mean_value = data[column].mean()
        q1_value = data[column].quantile(0.25)

        # Initialize variables to keep track of continuous duration
        continuous_duration = 0
        max_continuous_duration = 0

        if threshold == 'median':
            threshold_value = median_value
        elif threshold == 'mean':
            threshold_value = mean_value
        elif threshold == 'q1':
            threshold_value = q1_value

        # Iterate through the data points in the column
        for value in data[column]:
            if value > threshold_value:
                continuous_duration += 1
            else:
                max_continuous_duration = max(max_continuous_duration, continuous_duration)
                continuous_duration = 0

        # In case the last sequence extends to the end of the column
        max_continuous_duration = max(max_continuous_duration, continuous_duration)

        print("Max continuous duration above", threshold, "is:", max_continuous_duration)

duration_above(centroids_shifted, 'mean')
duration_above(centroids_shifted, 'median')   
duration_above(centroids_shifted, 'q1')   

# Clusters analysis 
# Read the CSV file into a DataFrame
file_path = '/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_week.csv'
df = pd.read_csv(file_path)

# Print the value counts of the 'Cluster' column
print(df['Cluster'].value_counts())

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