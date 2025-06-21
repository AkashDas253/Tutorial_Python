## **Filter Methods in Feature Selection**  

### **Overview**  
Filter methods select features based on statistical measures, independent of machine learning models. They evaluate feature relevance using variance, correlation, and statistical tests.  

---

## **1. Variance Threshold**  
**Usage**: Removes features with low variance, assuming low-variance features contribute less information.  

### **Syntax**  
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # Removes features with variance < 0.01
X_new = selector.fit_transform(X)
```

### **Use Case**  
- Suitable for removing constant or near-constant features in high-dimensional datasets.  

---

## **2. SelectKBest**  
**Usage**: Selects the top `k` features based on statistical scores.  

### **Syntax**  
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)  # ANOVA F-test for classification
X_new = selector.fit_transform(X, y)
```

### **Supported Statistical Tests**  

| **Test**      | **Use Case** |
|--------------|-------------|
| `chi2`       | Categorical data |
| `f_classif`  | ANOVA F-test for classification |
| `mutual_info_classif` | Mutual information for classification |
| `f_regression` | ANOVA F-test for regression |
| `mutual_info_regression` | Mutual information for regression |

---

## **3. SelectPercentile**  
**Usage**: Selects the top features based on a percentile ranking.  

### **Syntax**  
```python
from sklearn.feature_selection import SelectPercentile, chi2

selector = SelectPercentile(score_func=chi2, percentile=20)  # Top 20% features
X_new = selector.fit_transform(X, y)
```

### **Use Case**  
- Useful when selecting features dynamically based on dataset characteristics.  

---

## **4. Correlation-Based Selection (Feature Selection via Correlation Matrix)**  
**Usage**: Removes highly correlated features to avoid redundancy.  

### **Syntax**  
```python
import numpy as np
import pandas as pd

corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]  # Threshold = 0.9
X_new = pd.DataFrame(X).drop(columns=to_drop)
```

### **Use Case**  
- Helps in reducing multicollinearity in regression problems.  

---

## **Choosing the Right Filter Method**  

| **Scenario** | **Recommended Method** |
|-------------|------------------------|
| High-dimensional datasets | **VarianceThreshold** |
| Selecting top-ranked features | **SelectKBest, SelectPercentile** |
| Reducing multicollinearity | **Correlation-Based Selection** |

Filter methods are computationally efficient and useful for preprocessing before model training, ensuring relevant features are retained.