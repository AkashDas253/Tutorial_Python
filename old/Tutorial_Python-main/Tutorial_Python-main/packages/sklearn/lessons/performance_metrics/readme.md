## **Performance Metrics in Scikit-Learn**  

### **Overview**  
Performance metrics evaluate the effectiveness of machine learning models. Scikit-Learn provides a variety of metrics for classification, regression, and clustering models.

---

### **Classification Metrics**  

| Metric                 | Description |
|------------------------|-------------|
| **Accuracy**           | Measures the proportion of correctly predicted labels. |
| **Precision**          | Measures the proportion of true positives among predicted positives. |
| **Recall (Sensitivity)** | Measures the proportion of actual positives correctly identified. |
| **F1-Score**           | Harmonic mean of precision and recall, useful for imbalanced data. |
| **ROC-AUC Score**      | Measures the area under the ROC curve, evaluating classifier performance at different thresholds. |
| **Log Loss**           | Evaluates how well probability predictions match actual labels. |
| **Cohen’s Kappa**      | Measures agreement between predictions and actual labels beyond chance. |

#### **Syntax for Classification Metrics**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

y_true = [0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1]
y_prob = [0.1, 0.9, 0.4, 0.3, 0.8, 0.9, 0.7]  # Probabilities for ROC-AUC and Log Loss

# Compute metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-Score:", f1_score(y_true, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_true, y_prob))
print("Log Loss:", log_loss(y_true, y_prob))
```

---

### **Regression Metrics**  

| Metric                 | Description |
|------------------------|-------------|
| **Mean Absolute Error (MAE)**  | Average absolute difference between actual and predicted values. |
| **Mean Squared Error (MSE)**   | Average squared difference between actual and predicted values. |
| **Root Mean Squared Error (RMSE)** | Square root of MSE, interpretable in original units. |
| **R² Score (Coefficient of Determination)** | Measures how well predictions match actual values (1 = perfect, 0 = no relation). |
| **Mean Absolute Percentage Error (MAPE)** | Measures average percentage error. |

#### **Syntax for Regression Metrics**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = [3.2, 2.8, 4.1, 5.0, 6.2]
y_pred = [3.0, 3.1, 4.0, 4.8, 6.0]

# Compute metrics
print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
print("R² Score:", r2_score(y_true, y_pred))
```

---

### **Clustering Metrics**  

| Metric                 | Description |
|------------------------|-------------|
| **Adjusted Rand Index (ARI)** | Measures similarity between predicted and true clusters. |
| **Normalized Mutual Information (NMI)** | Evaluates shared information between clusters. |
| **Silhouette Score** | Measures how well each sample fits into its assigned cluster. |

#### **Syntax for Clustering Metrics**
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)

# Cluster using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# Compute metrics
print("ARI:", adjusted_rand_score(y_true, y_pred))
print("NMI:", normalized_mutual_info_score(y_true, y_pred))
print("Silhouette Score:", silhouette_score(X, y_pred))
```

---

### **Choosing the Right Metric**  

| Problem Type  | Recommended Metric |
|--------------|------------------|
| **Binary Classification** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Imbalanced Classification** | Precision, Recall, F1-Score, ROC-AUC |
| **Multi-Class Classification** | Accuracy, Precision-Recall per class, Log Loss |
| **Regression** | MAE, MSE, RMSE, R² |
| **Clustering** | ARI, NMI, Silhouette Score |

---