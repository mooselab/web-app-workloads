import os
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Specify the folder path
wikipedia = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/wikipedia'
wc98 = '/Users/roozbeh/Documents/benchmark_code/files/output_resample_1h/wc98'

def concat_df(path):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through files in the folder
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            # Read each CSV file and append it to the list
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Combine all DataFrames into one
    df_hour = pd.concat(dfs, ignore_index=True)
    df_hour.columns = ['Date', 'Value']

    # Convert the 'Date' column to datetime if it's not already
    df_hour['Date'] = pd.to_datetime(df_hour['Date'])

    # Sort and reset index
    df_hour = df_hour.sort_values(by=['Date'])
    df_hour = df_hour.reset_index(drop=True)

    # Group the data by day and sum the values
    df_day = df_hour.groupby(df_hour['Date'].dt.date)['Value'].sum().reset_index()

    return df_day, df_hour

df_wc98_day, df_wc98_hour = concat_df(wc98)
df_wikipedia_day, df_wikipedia_hour = concat_df(wikipedia)

# Function to format y-axis labels in millions
def format(x, pos):
    return f'{x / 1e6:.0f}M'

def format_hour(x, pos):
    return f'{x / 1e3:.0f}K'

# Set global font properties
rcParams['font.family'] = 'Times New Roman'  
rcParams['font.size'] = 40 


def show_daily(data, path):
    plt.rc('axes', labelsize=34)  
    plt.rc('xtick', labelsize=28)  
    plt.rc('ytick', labelsize=28)  

    # Create the line plots
    plt.figure(figsize=(16, 6))
    plt.plot(data['Date'], data['Value'], linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.tick_params(axis='x', labelrotation=45)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xlim(data['Date'].min(), data['Date'].max())

    if 'wikipedia' in path:
        plt.ylim(400000000, 720000000)

    plt.tight_layout()

    # Save the plot
    plt.savefig(path, format='pdf', bbox_inches='tight')

    # Display the first plot
    plt.show()

# Group by 'Date' and sum the 'Value' column and select a single day for plotting - wc98
df_wc98_hour = df_wc98_hour.groupby('Date', as_index=False)['Value'].sum()
selected_day_wc98 = df_wc98_hour.loc[df_wc98_hour['Date'].dt.date == pd.to_datetime('1998-06-20').date()]
selected_day_wc98 = selected_day_wc98.copy()
selected_day_wc98['Hour'] = selected_day_wc98['Date'].dt.hour

# Group by 'Date' and sum the 'Value' column and select a single day for plotting - wikipedia
df_wikipedia_hour = df_wikipedia_hour.groupby('Date', as_index=False)['Value'].sum()
selected_day_wikipedia = df_wikipedia_hour.loc[df_wikipedia_hour['Date'].dt.date == pd.to_datetime('2020-06-20').date()]
selected_day_wikipedia = selected_day_wikipedia.copy()
selected_day_wikipedia['Hour'] = selected_day_wikipedia['Date'].dt.hour

def show_hourly(data, path):
    plt.rc('axes', labelsize=40)  
    plt.rc('xtick', labelsize=34)  
    plt.rc('ytick', labelsize=34) 

    # Create the line plots
    plt.figure(figsize=(10, 5))
    plt.plot(data['Hour'], data['Value'], linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.tick_params(axis='x')
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

    if 'wc98' in path:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_hour))
    elif 'wikipedia' in path:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format))

    plt.tight_layout()

    # Save the plot
    plt.savefig(path, format='pdf', bbox_inches='tight')

    # Display the first plot
    plt.show()

show_daily(df_wikipedia_day, '/Users/roozbeh/Documents/Dars/PhD/Papers/My papers/Papers/paper3 - Workload_patterns/wikipedia_day.pdf')
show_daily(df_wc98_day, '/Users/roozbeh/Documents/Dars/PhD/Papers/My papers/Papers/paper3 - Workload_patterns/wc98_day.pdf')
show_hourly(selected_day_wc98, '/Users/roozbeh/Documents/Dars/PhD/Papers/My papers/Papers/paper3 - Workload_patterns/wc98_hour.pdf')
show_hourly(selected_day_wikipedia, '/Users/roozbeh/Documents/Dars/PhD/Papers/My papers/Papers/paper3 - Workload_patterns/wikipedia_hour.pdf')




# show_wikipedia_daily()
# show_wc98_daily()



# # Read the data from the CSV file
# file_path = '/Users/roozbeh/Documents/benchmark_code/clustered_results/clustered_kmeans_day.csv'
# df2 = pd.read_csv(file_path)
# df = df2.groupby(['File', 'Cluster']).size().unstack(fill_value=0).div(df2['File'].value_counts(), axis=0).fillna(0) * 100

# # Create a stacked bar plot
# ax = df.T.plot(kind='bar', stacked=True, figsize=(10, 6))

# # Set labels and title
# ax.set_xlabel('Files')
# ax.set_ylabel('Values')
# ax.set_title('Stacked Bar Plot of Files in Clusters')

# # Rotate x-axis labels for readability
# plt.xticks(rotation=0)

# # Add a legend
# ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Show the plot
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample DataFrame
# data = {
#     'Cluster': [0, 1, 2, 3],
#     'bu': [0.000000, 0.641026, 60.472973, 0.000000],
#     'calgary': [0.415896, 0.641026, 26.689189, 52.589641],
#     'clarknet': [0.415896, 0.000000, 0.000000, 0.996016],
#     'epa': [0.000000, 0.000000, 0.337838, 0.199203],
#     'nasa': [0.646950, 0.000000, 0.337838, 8.764940],
#     'retailrocket': [0.000000, 88.461538, 0.337838, 0.000000],
#     'saskatchewan': [1.571165, 0.000000, 0.000000, 35.856574],
#     'sdsc': [0.000000, 0.000000, 0.000000, 0.199203],
#     'wc': [1.848429, 10.256410, 9.797297, 0.597610],
#     'wikipedia': [95.101664, 0.000000, 2.027027, 0.796813]
# }

# df = pd.DataFrame(data)

# # Set the 'Cluster' column as the index
# df.set_index('Cluster', inplace=True)

# # Create a stacked bar plot
# ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))

# # Set labels and title
# ax.set_xlabel('Cluster')
# ax.set_ylabel('Values')
# ax.set_title('Stacked Bar Plot of Files in Clusters')

# # Rotate x-axis labels for readability
# plt.xticks(rotation=0)

# # Add a legend
# ax.legend(title='Files', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Show the plot
# plt.tight_layout()
# plt.show()







