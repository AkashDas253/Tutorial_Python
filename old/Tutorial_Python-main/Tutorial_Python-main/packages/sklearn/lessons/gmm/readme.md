## **Gaussian Mixture Model (GMM) in Scikit-Learn**  

### **Overview**  
Gaussian Mixture Model (GMM) is a probabilistic clustering algorithm that assumes data is generated from a mixture of multiple Gaussian distributions. Unlike K-Means, GMM assigns soft cluster memberships, meaning each data point has a probability of belonging to multiple clusters.

---

## **Key Features of GMM**  

| Feature | Description |
|---------|------------|
| **Soft Clustering** | Each point has a probability of belonging to multiple clusters. |
| **Elliptical Clusters** | Unlike K-Means, clusters are not constrained to be spherical. |
| **Expectation-Maximization (EM) Algorithm** | Iteratively refines cluster assignments based on likelihood maximization. |
| **Works Well with Overlapping Clusters** | Can model datasets where clusters have shared boundaries. |

---

## **Syntax for GMM in Scikit-Learn**  

```python
from sklearn.mixture import GaussianMixture

clf = GaussianMixture(
    n_components=3,    # Number of clusters (Gaussian distributions)
    covariance_type='full',  # Type of covariance matrix
    max_iter=100,       # Maximum iterations
    tol=1e-3,           # Convergence threshold
    random_state=42     # Seed for reproducibility
)
clf.fit(X)
labels = clf.predict(X)   # Get cluster assignments
probs = clf.predict_proba(X)  # Get probability of each point in all clusters
```

---

## **Parameters in `GaussianMixture`**  

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_components` | Number of Gaussian distributions (clusters). | `1` |
| `covariance_type` | Type of covariance matrix (`full`, `tied`, `diag`, `spherical`). | `full` |
| `tol` | Convergence threshold for stopping iterations. | `1e-3` |
| `max_iter` | Maximum number of iterations for the EM algorithm. | `100` |
| `random_state` | Random seed for reproducibility. | `None` |
| `n_init` | Number of initializations with different starting points. | `1` |
| `reg_covar` | Regularization added to covariance matrices for stability. | `1e-6` |

---

## **Covariance Types in GMM**  

| Covariance Type | Description | When to Use |
|-----------------|------------|-------------|
| **full** | Each cluster has its own covariance matrix. | Best flexibility, handles ellipsoidal clusters. |
| **tied** | All clusters share a single covariance matrix. | If clusters have similar shape. |
| **diag** | Each cluster has a diagonal covariance matrix. | If clusters have independent features. |
| **spherical** | Each cluster has a single variance value. | If clusters are circular. |

---

## **Choosing the Number of Clusters (`n_components`)**  

- **Bayesian Information Criterion (BIC)**
- **Akaike Information Criterion (AIC)**  
- **Silhouette Score**  

### **Finding Optimal `n_components`**  
```python
import numpy as np
from sklearn.mixture import GaussianMixture

bic_scores = []
for k in range(1, 11):
    clf = GaussianMixture(n_components=k, random_state=42)
    clf.fit(X)
    bic_scores.append(clf.bic(X))

best_k = np.argmin(bic_scores) + 1  # Best number of clusters
```

---

## **Advantages of GMM**  

✅ **Soft clustering**: Assigns probabilities instead of hard labels.  
✅ **Works well with elliptical clusters**: More flexible than K-Means.  
✅ **Handles overlapping clusters**: Good for real-world data.  

## **Limitations of GMM**  

❌ **Sensitive to initialization**: Can converge to local optima.  
❌ **Computationally expensive**: Higher complexity than K-Means.  
❌ **Assumes Gaussian distribution**: May not work well for arbitrary shapes.  

---

## **When to Use GMM Over K-Means?**  

| Condition | Preferred Algorithm |
|-----------|---------------------|
| Well-separated, spherical clusters | **K-Means** |
| Clusters with varying shapes and sizes | **GMM** |
| Overlapping clusters with probabilistic assignment | **GMM** |
| Large datasets with simple structure | **K-Means** |

---