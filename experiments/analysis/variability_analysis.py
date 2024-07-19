import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Common parameters and functions
name_mapping = {
    'sdsc': '12',
    'epa': '11',
    'clarknet': '10',
    'madrid': '9',
    'youtube': '8',
    'nasa': '7',
    'wc': '6',
    'retailrocket': '5',
    'bu': '4',
    'saskatchewan': '3',
    'calgary': '2',
    'wikipedia': '1',
}

ordered_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['File'] = df['File'].replace(name_mapping)
    return df

def calculate_burstiness(df, df_columns):
    df_data = df.iloc[:, df_columns:]
    mean = df_data.mean(axis=1)
    std = df_data.std(axis=1)
    burstiness = (std - mean) / (std + mean)
    df['Burstiness'] = burstiness
    return df

def plot_boxplot(df, y_label, output_file, y_ticks, y_lim):
    fig, ax = plt.subplots(figsize=(8, 4))
    for position, file in enumerate(ordered_keys):
        if file in df['File'].values:
            boxplot_data = df[df['File'] == file][y_label].dropna()
            ax.boxplot(boxplot_data, positions=[position+1], vert=True, showfliers=False, widths=0.5,
                patch_artist=True, boxprops=dict(facecolor='teal', color='teal', linewidth=3),
                medianprops=dict(color='darkslategrey', linewidth=2),
                whiskerprops=dict(color='teal', linewidth=2),
                capprops=dict(color='teal', linewidth=2),
                meanprops=dict(marker='.', markerfacecolor='red', markeredgecolor='red'))

    plt.xlabel('Workload', fontsize=32)
    plt.ylabel(y_label, fontsize=32)
    plt.grid(axis='y')
    plt.fontfamily = 'Times New Roman'
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.yticks(y_ticks, [str(tick) for tick in y_ticks])
    plt.ylim(y_lim)  
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.show()

def burstiness_analysis(file_path):
    df = preprocess_data(file_path)

    if 'daily' in file_path:
        granularity = 'daily'
        df_columns = 2
    elif 'weekly' in file_path:
        granularity = 'weekly'
        df_columns = 3

    df = calculate_burstiness(df, df_columns)
    output_file = os.path.join(output_path, f'burstiness_{granularity}.pdf')
    plot_boxplot(df, 'Burstiness', output_file, [-1, -0.5, 0, 0.5, 1], (-1, 1))

def variability_analysis(file_path):
    df = preprocess_data(file_path)

    if 'daily' in file_path:
        granularity = 'daily'
        df_columns = 2
        smoothing_span = 12
    elif 'weekly' in file_path:
        granularity = 'weekly'
        df_columns = 3
        smoothing_span = 7

    df_data = df.iloc[:, df_columns:].values
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data_zscore = scaler.fit_transform(df_data)
    reshaped_data_zscore = np.squeeze(scaled_data_zscore, axis=-1)
    df_scaled_data_zscore = pd.DataFrame(reshaped_data_zscore)
    df_ema = df_scaled_data_zscore.ewm(span=smoothing_span, axis=1).mean()
    df_std = df_ema.std(axis=1).values
    df['Variability'] = df_std

    output_file = os.path.join(output_path, f'variance_{granularity}.pdf')
    plot_boxplot(df, 'Variability', output_file, [0, 0.2, 0.4, 0.6, 0.8, 1], (0, 1))


merged_dataset_daily = '/merged_datasets/merged_dataset_daily.csv'
merged_dataset_weekly = '/merged_datasets/merged_dataset_weekly.csv'

# Define the base directory for saving files
output_path = '/results'

variability_analysis(merged_dataset_daily)
variability_analysis(merged_dataset_weekly)
burstiness_analysis(merged_dataset_daily)
burstiness_analysis(merged_dataset_weekly)