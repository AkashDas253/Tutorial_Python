## Dimensionality Reduction in Scikit-Learn  

#### Overview  
Dimensionality reduction techniques reduce the number of features while preserving as much information as possible. Scikit-Learn provides multiple methods, primarily categorized into **feature extraction** and **feature selection**.

---

### Techniques  

| Method                  | Type              | Description |
|-------------------------|------------------|-------------|
| **PCA (Principal Component Analysis)** | Feature Extraction | Projects data onto a lower-dimensional space using eigenvectors of covariance matrix. |
| **Kernel PCA** | Feature Extraction | Extends PCA to nonlinear relationships using kernel functions. |
| **Incremental PCA** | Feature Extraction | Computes PCA incrementally for large datasets. |
| **Sparse PCA** | Feature Extraction | Introduces sparsity in PCA components. |
| **Truncated SVD** | Feature Extraction | Similar to PCA but works on sparse matrices without mean-centering. |
| **Factor Analysis** | Feature Extraction | Assumes data is generated from latent variables plus noise. |
| **ICA (Independent Component Analysis)** | Feature Extraction | Decomposes data into statistically independent components. |
| **LDA (Linear Discriminant Analysis)** | Feature Extraction | Reduces dimensionality while maximizing class separability. |
| **t-SNE (t-Distributed Stochastic Neighbor Embedding)** | Feature Extraction | Preserves local structure for visualization (not ideal for preprocessing). |
| **UMAP (Uniform Manifold Approximation and Projection)** | Feature Extraction | Preserves global and local structure better than t-SNE. |
| **Feature Selection (VarianceThreshold, SelectKBest, RFE, etc.)** | Feature Selection | Removes less important features based on criteria (e.g., variance, correlation, model importance). |

---

### Syntax with Parameters  

#### **1. PCA**
```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=None,    # Number of principal components to keep
    copy=True,            # Whether to copy data or overwrite
    whiten=False,         # Normalize variance in transformed data
    svd_solver='auto',    # Solver to use ('auto', 'full', 'arpack', 'randomized')
    tol=0.0,              # Convergence tolerance
    iterated_power='auto',# Number of power iterations for 'randomized' solver
    random_state=None     # Seed for reproducibility
)
X_reduced = pca.fit_transform(X)
```

#### **2. Kernel PCA**
```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=None,    # Number of components
    kernel='linear',      # Kernel type ('linear', 'poly', 'rbf', 'sigmoid', etc.)
    gamma=None,           # Kernel coefficient for ‘rbf’, ‘poly’, ‘sigmoid’
    degree=3,             # Degree for polynomial kernel
    coef0=1,              # Independent term in polynomial and sigmoid kernels
    alpha=1.0,            # Hyperparameter for regularization
    fit_inverse_transform=False,  # Whether to enable inverse transform
    eigen_solver='auto',  # Solver for eigenvalue decomposition
    tol=0,                # Convergence tolerance
    max_iter=None,        # Max iterations (if solver requires it)
    remove_zero_eig=False # Remove zero eigenvalues
)
X_reduced = kpca.fit_transform(X)
```

#### **3. Incremental PCA**
```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(
    n_components=None,    # Number of principal components
    batch_size=None       # Batch size for incremental processing
)
X_reduced = ipca.fit_transform(X)
```

#### **4. Truncated SVD**
```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(
    n_components=2,    # Number of singular values to keep
    algorithm='randomized',  # Solver type ('arpack' or 'randomized')
    n_iter=5,          # Number of iterations for randomized solver
    random_state=None  # Seed for reproducibility
)
X_reduced = svd.fit_transform(X)
```

#### **5. Factor Analysis**
```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(
    n_components=None, # Number of factors
    tol=0.01,          # Stopping tolerance
    copy=True,         # Whether to copy data
    max_iter=1000      # Maximum number of iterations
)
X_reduced = fa.fit_transform(X)
```

#### **6. Independent Component Analysis (ICA)**
```python
from sklearn.decomposition import FastICA

ica = FastICA(
    n_components=None,  # Number of independent components
    algorithm='parallel',  # Algorithm type ('parallel' or 'deflation')
    whiten=True,        # Whether to whiten the data
    fun='logcosh',      # Nonlinearity function
    max_iter=200,       # Maximum iterations
    tol=0.0001,         # Convergence tolerance
    random_state=None   # Seed for reproducibility
)
X_reduced = ica.fit_transform(X)
```

#### **7. LDA**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(
    n_components=None,  # Number of components
    solver='svd',       # Solver type ('svd', 'lsqr', 'eigen')
    shrinkage=None,     # Shrinkage parameter (None, ‘auto’, float)
    priors=None,        # Class priors
    tol=0.0001          # Convergence tolerance
)
X_reduced = lda.fit_transform(X, y)
```

#### **8. t-SNE**
```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,    # Number of dimensions
    perplexity=30,     # Trade-off between local and global structure
    early_exaggeration=12.0,  # How tight clusters are
    learning_rate=200, # Step size in optimization
    n_iter=1000,       # Number of iterations
    random_state=None  # Seed for reproducibility
)
X_reduced = tsne.fit_transform(X)
```

#### **9. UMAP**
```python
import umap

umap_reducer = umap.UMAP(
    n_components=2,      # Number of dimensions
    n_neighbors=15,      # Number of neighbors for local structure
    min_dist=0.1,        # Minimum distance between points
    metric='euclidean',  # Distance metric
    random_state=None    # Seed for reproducibility
)
X_reduced = umap_reducer.fit_transform(X)
```

---

### **Choosing the Right Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| Large datasets (>10,000 samples) | **Incremental PCA, Truncated SVD** |
| Linear correlation among features | **PCA, Truncated SVD** |
| Nonlinear relationships | **Kernel PCA, UMAP** |
| Interpretable factors needed | **Factor Analysis, LDA** |
| Features are independent signals | **ICA** |
| Data visualization (2D/3D) | **t-SNE, UMAP** |
| Sparse data | **Truncated SVD, Sparse PCA** |
| High-class separability required | **LDA** |

---