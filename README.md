# Understanding Web Application Workloads and Their Applications

This repository hosts the replication package for the paper "Understanding Web Application Workloads and Their Applications: Systematic Literature Review and Characterization."

## Abstract
Web applications, accessible via web browsers over the Internet, facilitate complex functionalities without local software installation. In the context of web applications, a workload refers to the number of user requests sent by users or applications to the underlying system. Existing studies have leveraged web application workloads to achieve various objectives, such as workload prediction and auto-scaling. However, these studies are conducted in an *ad hoc* manner, lacking a systematic understanding of the characteristics of web application workloads. In this study, we first conduct a systematic literature review to identify and analyze existing studies leveraging web application workloads. Our analysis sheds light on their workload utilization, analysis techniques, and high-level objectives. We further systematically analyze the characteristics of the web application workloads identified in the literature review. Our analysis centers on characterizing these workloads at two distinct temporal granularity: daily and weekly. We successfully identify and categorize three daily and three weekly patterns within the workloads. By providing a statistical characterization of these workload patterns, our study highlights the uniqueness of each pattern, paving the way for the development of realistic workload generation and resource provisioning techniques that can benefit a range of applications and research areas.

## Installation

To install and run the scripts, follow these steps:

```bash
git clone https://github.com/mooselab/web-app-workloads
cd web-app-workloads
pip install -r requirements.txt
```

## Usage Instructions
The repository is organized into five main folders:

1. articles: Information about the studied articles.
2. data: Contains 12 raw datasets that were extracted and used in this study:
- Wikipedia
- Worldcup98
- NASA
- Saskatchewan
- Calgary
- EPA
- ClarkNet
- Retailrocket
- Boston
- SDSC
- Youtube
- Madrid
3. merged_datasets: The raw datasets after the pre-processing steps and aggregated in two time granularities: daily and weekly.
4. experiments: This folder contains the code for conducting the experiments and analyzing the results. It is further divided into two subfolders:
- clustering: Handles the steps to cluster workloads.
- analysis: Contains code for the analysis of workloads to address our research questions.
5. results: This folder houses the clustered results of the workloads, which are instrumental in answering our research questions.

## License
This project is licensed under the MIT License.
