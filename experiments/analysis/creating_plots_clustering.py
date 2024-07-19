import os
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

plt_settings = {
    'font.size': 40,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'font.family': 'Times New Roman'
}

# Specify the folder path
wikipedia = '/data/wikipedia'
wc98 = '/data/wc98'

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

# select a single week for plotting - wc98
df_wc98_day['Date'] = pd.to_datetime(df_wc98_day['Date'])
selected_day_wc98 = df_wc98_day[(df_wc98_day['Date'] >= '1998-06-23') & (df_wc98_day['Date'] <= '1998-06-29')]


# select a single week for plotting - wikipedia
df_wikipedia_day['Date'] = pd.to_datetime(df_wikipedia_day['Date'])
selected_day_wikipedia = df_wikipedia_day[(df_wikipedia_day['Date'] >= '2023-01-01') & (df_wikipedia_day['Date'] <= '2023-01-07')]

# Function to format y-axis labels in millions
def format(x, pos):
    return f'{x / 1e6:.0f}M'

def format_hour(x, pos):
    return f'{x / 1e3:.0f}K'

def show_daily(data, path):
    plt.rc('axes', labelsize=40)  
    plt.rc('xtick', labelsize=34)  
    plt.rc('ytick', labelsize=34)  

    # Create the line plots
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Value'], linewidth=5, linestyle='-', color='teal')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.tick_params(axis='x')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xlim(data['Date'].min(), data['Date'].max())
    plt.grid(axis='y')
    plt.xticks(plt.xticks()[0], ['1', '2', '3', '4', '5', '6', '7']) 

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.show()

# Group by 'Date' and sum the 'Value' column and select a single day for plotting - wc98
df_wc98_hour = df_wc98_hour.groupby('Date', as_index=False)['Value'].sum()
selected_hour_wc98 = df_wc98_hour.loc[df_wc98_hour['Date'].dt.date == pd.to_datetime('1998-06-20').date()]
selected_hour_wc98 = selected_hour_wc98.copy()
selected_hour_wc98['Hour'] = selected_hour_wc98['Date'].dt.hour

# Group by 'Date' and sum the 'Value' column and select a single day for plotting - wikipedia
df_wikipedia_hour = df_wikipedia_hour.groupby('Date', as_index=False)['Value'].sum()
selected_hour_wikipedia = df_wikipedia_hour.loc[df_wikipedia_hour['Date'].dt.date == pd.to_datetime('2020-06-20').date()]
selected_hour_wikipedia = selected_hour_wikipedia.copy()
selected_hour_wikipedia['Hour'] = selected_hour_wikipedia['Date'].dt.hour

def show_hourly(data, path):
    plt.rc('axes', labelsize=40)  
    plt.rc('xtick', labelsize=34)  
    plt.rc('ytick', labelsize=34) 

    # Create the line plots
    plt.figure(figsize=(10, 5))
    plt.plot(data['Hour'], data['Value'], linewidth=5, linestyle='-', color='teal')
    plt.xticks([0, 6, 12, 18, 24], ['0', '6', '12', '18', '24'])
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.tick_params(axis='x')
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.grid(axis='y')

    if 'wc98' in path:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_hour))
    elif 'wikipedia' in path:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format))

    plt.tight_layout()

    # Save the plot
    plt.savefig(path, format='pdf', bbox_inches='tight')

    # Display the first plot
    plt.show()

show_daily(selected_day_wikipedia, '/result/wikipedia_day.pdf')
show_daily(selected_day_wc98, '/result/wc98_day.pdf')
show_hourly(selected_hour_wc98, '/result/wc98_hour.pdf')
show_hourly(selected_hour_wikipedia, '/result/wikipedia_hour.pdf')

# Read the data from the CSV file
df = pd.read_csv('/results/clustered_kmeans_day.csv')

# Extract the day of the week
df['Date'] = pd.to_datetime(df['Date'])

# Determine whether each day is a weekday or weekend
df['WeekdayOrWeekend'] = df['Day'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

# Data grouping
grouped = df.groupby(['Cluster', 'WeekdayOrWeekend']).size().unstack(fill_value=0)

# Transpose the DataFrame
grouped = grouped.T

# Calculate the percentage 
grouped['Total'] = grouped.sum(axis=1)

grouped[0] = grouped[0] / grouped['Total'] * 100
grouped[1] = grouped[1] / grouped['Total'] * 100
grouped[2] = grouped[2] / grouped['Total'] * 100

# Modify the DataFrame
cluster_names = {0: 'D1', 1: 'D2', 2: 'D3'}
grouped.rename(columns=cluster_names, inplace=True)
grouped = grouped.drop('Total', axis=1)

# Plot the results
with plt.rc_context(rc=plt_settings):
    ax = grouped.plot(kind='bar', width=0.22, stacked=True, figsize=(9,8), color=['teal', 'gold', 'coral'])
    plt.xlabel('')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend(loc='upper center', fontsize=28, bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=3)
    plt.yticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100'])

    plt.tight_layout()
    file_name = f'/results/daily_week.pdf'
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    plt.show()

    # Read the data from the new CSV file
df_week = pd.read_csv('results/clustered_kmeans_week.csv')

# Extract the month of the year
df_week['Date'] = pd.to_datetime(df_week['Date'])
df_week['Month'] = df_week['Date'].dt.month

# Define function to map month to quarter
def map_month_to_quarter(month):
    if month in [1, 2, 3]:
        return 'Spring'
    elif month in [4, 5, 6]:
        return 'Summer'
    elif month in [7, 8, 9]:
        return 'Fall'
    elif month in [10, 11, 12]:
        return 'Winter'
    else:
        print('Invalid month:', month)

# Determine the quarter for each day
df_week['Quarter'] = df_week['Month'].apply(map_month_to_quarter)

# Data grouping
grouped_week = df_week.groupby(['Cluster', 'Quarter']).size().unstack(fill_value=0)

# Transpose the DataFrame
grouped_week = grouped_week.T

# Calculate the percentage
grouped_week['Total'] = grouped_week.sum(axis=1)
grouped_week = grouped_week.div(grouped_week['Total'], axis=0) * 100
grouped_week.drop(columns='Total', inplace=True)

# Reorder the columns
grouped_week = grouped_week.reindex(index=['Spring', 'Summer', 'Fall', 'Winter'])

# Rename the clusters
cluster_names = {0: 'W1', 1: 'W2', 2: 'W3'}
grouped_week.rename(columns=cluster_names, inplace=True)
print(grouped_week)

# Plot the results
with plt.rc_context(rc=plt_settings):
    ax = grouped_week.plot(kind='bar', stacked=True, figsize=(9, 8), color=['teal', 'gold', 'coral'])
    plt.xlabel('')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend(loc='upper center', fontsize=28, bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=3)
    plt.yticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100'])

    plt.tight_layout()
    file_name = f'/result/weekly_quarter.pdf'
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    plt.show()