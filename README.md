# OPTIMAL-EM: A Software Tool for Optimised Web Accessibility Evaluation

The OPTIMAL EM Tool is designed to support research in web accessibility evaluation. Currently, the tool calculates optimal sample sizes for more efficient manual audits and outputs a prototype markdown report of the findings. 

## Overview

This tool processes a collection of HTML files to determine representative pages for optimised web accessibility evaluations. By clustering web pages based on their structural (or content) similarities and calculating complexity variances, the tool identifies representative samples that auditors should manually review.

## Features

- Clustering: Uses t-SNE for dimensionality reduction and DBSCAN for clustering web pages.
- Complexity Analysis: Calculates the complexity variance within each cluster.
- Reporting: Generates a markdown report summarising the findings and recommendations.

## Usage

### Prepare HTML Files

Place your HTML files in the res/res-*/ directory. You can change this path in `constant.py` if needed.

### Configure Parameters
Review and adjust parameters (e.g., clustering parameters, sample size calculations) in `constant.py` to suit the dataset.

### Run the Tool
Run the main script:

```bash
python main.py
```

### View the Results
A markdown report will be generated and a CSV file `output.csv` will contain details of the sampled pages.

## Code Structure

The code was developed ad-hoc throught the PhD and is in need of significant refactoring. [Please see this tool for a more optimised script](https://github.com/alexhambley/OPTIMAL-EM-Complexity-Pipeline).

### `main.py` 
The entry point of the application. It orchestrates the clustering, complexity analysis, sample size calculation, and report generation.

### `cluster.py`
Handles the clustering of web pages:

- `run_dbscan_cluster_functions()`: Performs feature extraction, dimensionality reduction using t-SNE, and clustering with DBSCAN.
- `check_if_k_means_required()`: Determines if K-Means clustering is needed for further clustering a dataset following DBSCAN. Rarely necessasary but available if further grouping is needed. 


### `feature_extraction.py`
Extracts features from HTML files for clustering. e.g. `get_html_block_structure(soup)` wil extract the block-level HTML tags from a BeautifulSoup object.

### `complexity.py`
Calculates complexity metrics for each HTML file to assess the variance within clusters.

### `samplesize.py`
Contains functions related to sample size calculation:

- `get_sample_size_from_z_score()`: Computes the sample size based on the desired confidence level and population size.
- `get_weighted_sample()`: Determines the number of samples to draw from each cluster.
