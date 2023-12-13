# Characterizing Workload Patterns

This repository hosts the replication package for the paper "Characterizing the Workload Patterns of Web Applications."

## Introduction

The repository is organized into three main folders:

1. **Experiments:** This folder encompasses the code for conducting the experiments. It is further divided into three subfolders: `extracting_traces`, `clustering`, and `analysis`. The first subfolder is utilized for extracting traces (available in the [Extracted Traces Folder](./extracted_traces/)). The second subfolder handles the clustering of traces, and the results can be found in the [Results Folder](./results/). The third subfolder contains code for the analysis of traces to address our research questions.

2. **Extracted Traces:** This folder stores the raw data of the 10 traces studied in our research.

3. **Results:** This folder houses the clustered results of the traces, which are instrumental in answering our research questions.

## Install

To install and run the scripts, follow these steps:

```bash
git clone https://github.com/workloadPatterns/characterizing_workload_patterns
pip install -r requirements.txt
```

## Experiments

The `experiments` folder contains code for our primary experiments. Specific scripts include:

- `clustering_kmeans_silhouette_day.py`: Performs clustering on trace data at the daily granularity.
- `clustering_kmeans_silhouette_week.py`: Performs clustering on trace data at the weekly granularity.
- `clusters_analysis.py`: Conducts analysis on clusters and cluster centroids after the clustering process.
- `correlation_analysis.py`: Carries out correlation analysis between daily and weekly clusters.
- `creating_plots.py`: Generates figures used in the article.
- `extracting_traces` folder: Contains code for extracting the 10 traces used in our study.