# Statistical Learning Project Report
This report is regarding to the project for Statistical Learning Course in Shahid Beheshti University, Faculty of 
Mathematics and Computer Science. This report covers the following contents:

## Data
To learn about the data set please follow these links.
 
- <a href="https://github.com/ArefMq/ML-Project/blob/master/dataset/winequality.txt">Dataset Description</a>
- <a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality">Dataset Web site</a>

Dataset for clustering is:
- <a href="https://archive.ics.uci.edu/ml/datasets/BuddyMove+Dataset">Clustering Dataset Web site</a>
 
## Results
To view result please follow these links:
 
- <a href="https://github.com/ArefMq/ML-Project/blob/master/regression/README.md">Regression Results (project #1)</a>
- <a href="https://github.com/ArefMq/ML-Project/blob/master/classification/README.md">Classification Results (project #2)</a>
- <a href="https://github.com/ArefMq/ML-Project/blob/master/svm/README.md">SVM Results (project #3)</a>
- <a href="https://github.com/ArefMq/ML-Project/blob/master/clustering/README.md">Clustering Results (project #4)</a>
- <a href="https://github.com/ArefMq/ML-Project/blob/master/nn/README.md">Neural Network Results (project #5)</a>

------------------------------------------------------------------------------------------------------------------------

## Requirements
This project is developed on Python. In order it you will need the following python-packages installed. You can download 
and install them via `pip` tool.
- numpy
- sklearn
- scipy
- metrics
- pandas
- keras (for running NN)
- tensorflow (for running NN)

## Runing Project
In order to run the project, inside the project root folder call the run.py with a project name argument. By running each project it will run and illustrate its results.
 
 ```bash
python run.py PROJECT_NAME
 ``` 
 Where `PROJECT_NAME` could be one of these:
 
 - `classification` or `cl` for running Classification
 - `bench-mark` or `bm` for running Bench-Mark
 - `feature-selection` or `fs` for running Feature Selection on regression data
 - `linear-regression` or `lr` for running Linear Regression
 - `reg-all` for running all Regression based modules
 - `kmeans` or `km` for running k-means clustering
 - `hierarchical-clustering` or `hc` for running Hierarchical Clustering (Agglomerative-Clustering)
 - `neural-networks` or `nn` for running Neural Network Classification

You can also run these commands for running projects based on the part they’re described in:

 - `p1` for running part #1 (Regression)
 - `p2` for running part #2 (Classification)
 - `p3` for running part #3 (SVM)
 - `p4` for running part #4 (Clustering)
 - `p5` for running part #5 (Neural Network)

## Report
In order to read more about this project please follow this link:
https://github.com/ArefMq/ML-Project/blob/master/report.pdf

