# Clustering
This part, contains project sample for Classification in Python using Sci-Kit tools.

## Run Clustering
In order to run the project please follow <a href="https://github.com/ArefMq/ML-Project/blob/master/README.md#Runing Project">
this link</a>.

## Dataset
Data set for this section is different, please visit the dataset here: 
<a href="https://archive.ics.uci.edu/ml/datasets/BuddyMove+Dataset">Clustering Dataset Web site</a>
 
## Results
The goal is to group people by the comments they have made on social medias. This goal is achieved by using *K-Means* 
Clustering and *Hierarchical-Clustering*. And then, the result are compared together via *Silhouette Score*.
As the results shows, 3 class worked best in both cases. However, the K-Means has higher score.

K-Means:

| N-Cluster (K) | Silhouette Score |
|---------------|------------------|
| 2             | 0.31             |
| 3             | 0.35             |
| 4             | 0.34             |
| 5             | 0.29             |
| 6             | 0.30             |
| 7             | 0.27             |
| 8             | 0.31             |
| 9             | 0.31             |
| best k = 3    | 0.35             |


Hierarchical-Clustering:

| N-Cluster     | Silhouette Score |
|---------------|------------------|
| 2             | 0.24             |
| 3             | 0.29             |
| 4             | 0.25             |
| 5             | 0.25             |
| 6             | 0.25             |
| 7             | 0.27             |
| 8             | 0.29             |
| 9             | 0.28             |
| best k = 3    | 0.29             |


This is PCA plot of the clustered data for K-Means Clustering:
![alt text][kmeans]
This is PCA plot of the clustered data for Hierarchical Clustering:
![alt text][hc]

[kmeans]: https://github.com/ArefMq/ML-Project/blob/master/clustering/kmeans.png "K-Means"
[hc]: https://github.com/ArefMq/ML-Project/blob/master/clustering/kmeans.png "Hierarchical-Clustering"


