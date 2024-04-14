import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# # Load the dataset
path = '/Users/roozbeh/Documents/Dars/PhD/Papers/my_papers/Papers/paper3_Workload_patterns/'
df = pd.read_csv(path + 'workloads_years.csv')

# Extract Paper Year and Dataset Year
paper_years = df['Paper Year']
dataset_years = df['Dataset Year']

print('Paper years median:', paper_years.median(), ', Dataset years median:', dataset_years.median())
print()

# Filter rows where Paper Year is 2015 or greater
filtered_df = df[df['Paper Year'] >= 2015]
print('Frequency of dataset utilization for papers published after 2015')
print(filtered_df['Dataset Year'].value_counts())

# Add jitter or random noise to the data points
jitter = 0.3
paper_years_jittered = paper_years + np.random.uniform(-jitter, jitter, len(paper_years))
dataset_years_jittered = dataset_years + np.random.uniform(-jitter, jitter, len(dataset_years))

# # Set global font properties
rcParams['font.family'] = 'Times New Roman'  

# Plot the scatter plot
plt.figure(figsize=(7, 3))
plt.rc('font', size=15)
plt.rc('axes', labelsize=17)  
plt.rc('xtick', labelsize=12)  
plt.rc('ytick', labelsize=12)  
plt.scatter(paper_years_jittered, dataset_years_jittered, color='blue', alpha=0.5)
plt.xlabel('Article Publication Year')
plt.ylabel('Dataset Year')
plt.grid(True)
plt.tight_layout()
plt.savefig(path + 'workload_years.pdf', bbox_inches='tight')
plt.show()

# Group the data by objectives and publication years
grouped_data = df.groupby(['Objectives', 'Paper Year']).size().reset_index(name='Count')

# Calculate the cumulative count of papers for each objective and publication year
grouped_data['Cumulative Count'] = grouped_data.groupby('Objectives')['Count'].cumsum()

# Plot the line graph
plt.figure(figsize=(7, 3))
plt.rc('font', size=12)
plt.rc('axes', labelsize=17)  
plt.rc('xtick', labelsize=12)  
plt.rc('ytick', labelsize=12)
for objective, group in grouped_data.groupby('Objectives'):
    plt.plot(group['Paper Year'], group['Cumulative Count'], marker='o', label=objective)

plt.xticks([1995, 2000, 2005, 2010, 2015, 2020, 2025], ['1995', '2000', '2005', '2010', '2015', '2020', '2025'])
plt.xlabel('Year')
plt.ylabel('Papers')
plt.legend(loc=2, ncol=2)
plt.grid(True)
plt.savefig(path + 'objective_years.pdf', bbox_inches='tight')
plt.show()