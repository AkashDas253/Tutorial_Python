## Clustering
Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups.

#### Syntax

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# KMeans
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, algorithm='auto')
kmeans.fit(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
dbscan.fit(X)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
agglo.fit(X)
```

#### Parameters
- **`n_clusters=8`**: The number of clusters to form as well as the number of centroids to generate. {integer}
- **`init='k-means++'`**: Method for initialization. {‘k-means++’, ‘random’, ndarray, callable}
- **`n_init=10`**: Number of time the k-means algorithm will be run with different centroid seeds. {integer}
- **`max_iter=300`**: Maximum number of iterations of the k-means algorithm for a single run. {integer}
- **`tol=1e-4`**: Relative tolerance with regards to inertia to declare convergence. {float}
- **`precompute_distances='deprecated'`**: Precompute distances (faster but takes more memory). {‘auto’, True, False}
- **`verbose=0`**: Verbosity mode. {integer}
- **`random_state=None`**: Determines random number generation for centroid initialization. {None, integer}
- **`copy_x=True`**: When pre-computing distances it is more numerically accurate to center the data first. {True, False}
- **`algorithm='auto'`**: K-means algorithm to use. {‘auto’, ‘full’, ‘elkan’}
- **`eps=0.5`**: The maximum distance between two samples for one to be considered as in the neighborhood of the other. {float}
- **`min_samples=5`**: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. {integer}
- **`metric='euclidean'`**: The metric to use when calculating distance between instances in a feature array. {string, callable}
- **`metric_params=None`**: Additional keyword arguments for the metric function. {dict}
- **`leaf_size=30`**: Leaf size passed to BallTree or cKDTree. {integer}
- **`affinity='euclidean'`**: Metric used to compute the linkage. {string, callable}
- **`memory=None`**: Used to cache the output of the computation of the tree. {None, string, object}
- **`connectivity=None`**: Connectivity matrix. {array-like, callable, None}
- **`compute_full_tree='auto'`**: Stop early the construction of the tree at n_clusters. {‘auto’, True, False}
- **`linkage='ward'`**: Which linkage criterion to use. {‘ward’, ‘complete’, ‘average’, ‘single’}
- **`distance_threshold=None`**: The linkage distance threshold above which, clusters will not be merged. {float, None}

#### Attributes
- **`cluster_centers_`**: Coordinates of cluster centers.
- **`labels_`**: Labels of each point.
- **`inertia_`**: Sum of squared distances of samples to their closest cluster center.
- **`n_iter_`**: Number of iterations run.
- **`core_sample_indices_`**: Indices of core samples.
- **`components_`**: Copy of each core sample found by training.
- **`n_features_in_`**: Number of features seen during fit.
- **`feature_names_in_`**: Names of features seen during fit.
- **`n_clusters_`**: The number of clusters found by the algorithm.
- **`n_leaves_`**: Number of leaves in the hierarchical tree.
- **`n_connected_components_`**: The estimated number of connected components in the graph.
- **`children_`**: The children of each non-leaf node.

#### Functions
- **`fit(X, y=None)`**: Compute clustering.
- **`fit_predict(X, y=None)`**: Compute cluster centers and predict cluster index for each sample.
- **`fit_transform(X, y=None)`**: Compute clustering and transform X to cluster-distance space.
- **`predict(X)`**: Predict the closest cluster each sample in X belongs to.
- **`transform(X)`**: Transform X to a cluster-distance space.