## **Unsupervised Learning in Scikit-Learn**  

### **Overview**  
Unsupervised learning finds patterns in data without labeled outputs. It is used for clustering, dimensionality reduction, anomaly detection, and association rule learning.

---

### **Types of Unsupervised Learning Models**  

| Model Type | Description | Best Use Case |
|------------|-------------|--------------|
| **Clustering** | Groups data based on similarity. | Customer segmentation, anomaly detection. |
| **Dimensionality Reduction** | Reduces feature space while preserving information. | Feature selection, noise reduction. |
| **Anomaly Detection** | Identifies rare events in data. | Fraud detection, network security. |
| **Association Rule Learning** | Finds relationships between variables. | Market basket analysis, recommendation systems. |

---

## **1. Clustering Algorithms**  

### **1.1 K-Means Clustering**  
**Usage**: Groups data into `k` clusters by minimizing variance.  
**Syntax**:  
```python
from sklearn.cluster import KMeans

clf = KMeans(
    n_clusters=3,      # Number of clusters
    init='k-means++',  # Initialization method
    max_iter=300,      # Maximum number of iterations
    random_state=42    # Random seed for reproducibility
)
clf.fit(X)
labels = clf.predict(X)
```

---

### **1.2 DBSCAN (Density-Based Clustering)**  
**Usage**: Identifies clusters based on density and detects outliers.  
**Syntax**:  
```python
from sklearn.cluster import DBSCAN

clf = DBSCAN(
    eps=0.5,          # Maximum distance between points in a cluster
    min_samples=5     # Minimum samples in a neighborhood
)
clf.fit(X)
labels = clf.labels_
```

---

### **1.3 Hierarchical Clustering (Agglomerative Clustering)**  
**Usage**: Forms a hierarchy of clusters using linkage methods.  
**Syntax**:  
```python
from sklearn.cluster import AgglomerativeClustering

clf = AgglomerativeClustering(
    n_clusters=3,      # Number of clusters
    linkage='ward'     # Linkage criterion: 'ward', 'complete', 'average'
)
clf.fit(X)
labels = clf.labels_
```

---

## **2. Dimensionality Reduction**  

### **2.1 Principal Component Analysis (PCA)**  
**Usage**: Reduces dimensions while preserving variance.  
**Syntax**:  
```python
from sklearn.decomposition import PCA

clf = PCA(
    n_components=2    # Number of principal components
)
X_reduced = clf.fit_transform(X)
```

---

### **2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)**  
**Usage**: Projects high-dimensional data into 2D/3D for visualization.  
**Syntax**:  
```python
from sklearn.manifold import TSNE

clf = TSNE(
    n_components=2,   # Output dimensions
    perplexity=30,    # Balance between local/global aspects
    random_state=42   # Random seed for reproducibility
)
X_reduced = clf.fit_transform(X)
```

---

### **2.3 Independent Component Analysis (ICA)**  
**Usage**: Separates independent sources from mixed signals.  
**Syntax**:  
```python
from sklearn.decomposition import FastICA

clf = FastICA(
    n_components=2    # Number of independent components
)
X_reduced = clf.fit_transform(X)
```

---

## **3. Anomaly Detection**  

### **3.1 Isolation Forest**  
**Usage**: Detects anomalies by isolating data points.  
**Syntax**:  
```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(
    contamination=0.1, # Proportion of outliers
    random_state=42    # Random seed for reproducibility
)
clf.fit(X)
anomalies = clf.predict(X)
```

---

### **3.2 One-Class SVM**  
**Usage**: Identifies outliers in high-dimensional data.  
**Syntax**:  
```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(
    nu=0.1,  # Proportion of anomalies
    kernel='rbf'
)
clf.fit(X)
anomalies = clf.predict(X)
```

---

## **4. Association Rule Learning**  

### **4.1 Apriori Algorithm**  
**Usage**: Finds frequent itemsets for association rules.  
**Syntax**:  
```python
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(
    df, 
    min_support=0.5,   # Minimum support threshold
    use_colnames=True
)
```

---

### **Choosing the Right Unsupervised Learning Model**  

| Scenario | Recommended Model |
|----------|--------------------|
| Identifying clusters in structured data | **K-Means** |
| Finding clusters with varying density | **DBSCAN** |
| Clustering with hierarchical relationships | **Agglomerative Clustering** |
| Reducing dimensionality for visualization | **PCA, t-SNE** |
| Detecting anomalies in transactional data | **Isolation Forest** |
| Finding frequent itemsets in transactions | **Apriori Algorithm** |

---