# Characterizing Web Application Workload Patterns

This repository hosts the replication package for the paper "Understanding Web Application Workloads and Their Applications: Systematic Literature Review and Characterization."

## Introduction

The repository is organized into three main folders:

1. **Experiments:** This folder encompasses the code for conducting the experiments. It is further divided into three subfolders: `extracting_traces`, `clustering`, and `analysis`. The first subfolder is utilized for extracting workloads (available in the [extracted_traces](./extracted_traces/) folder). The second subfolder handles the clustering of workloads, and the results can be found in the [results](./results/) folder. The third subfolder contains code for the analysis of workloads to address our research questions.

2. **Extracted Traces:** This folder stores the raw data of the 12 workloads studied in our research.

3. **Results:** This folder houses the clustered results of the workloads, which are instrumental in answering our research questions.

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
- `clustering_analysis.py`: Conducts analysis on clusters after the clustering process.
- `correlation_analysis.py`: Carries out correlation analysis between daily and weekly clusters.
- `creating_plots_clustering.py`: Generates figures used in the first research question.
- `creating_plots_slr.py`: Generates figures used in the second research question.
- `extracting_traces` folder: Contains code for extracting the 12 workloads used in our study.