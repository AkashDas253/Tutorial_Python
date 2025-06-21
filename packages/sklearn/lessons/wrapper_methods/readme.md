## **Wrapper Methods in Feature Selection**  

### **Overview**  
Wrapper methods evaluate feature subsets by training and testing a machine learning model iteratively. They find the best feature combination by assessing model performance. These methods are computationally expensive but often yield better results than filter methods.  

---

## **1. Recursive Feature Elimination (RFE)**  
**Usage**: Iteratively removes the least important features based on a model’s coefficients or feature importance.  

### **Syntax**  
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
selector = RFE(model, n_features_to_select=5)  # Selects top 5 features
X_new = selector.fit_transform(X, y)
```

### **Use Case**  
- Best for models with feature importance attributes (`coef_` or `feature_importances_`).  
- Works well with linear models and tree-based classifiers.  

---

## **2. Recursive Feature Elimination with Cross-Validation (RFECV)**  
**Usage**: Similar to RFE but uses cross-validation to select the optimal number of features.  

### **Syntax**  
```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
selector = RFECV(model, cv=5)  # 5-fold cross-validation
X_new = selector.fit_transform(X, y)
```

### **Use Case**  
- Useful when the best number of features is unknown.  
- Prevents overfitting by selecting an optimal feature subset.  

---

## **3. Sequential Feature Selection (SFS)**  
**Usage**: Selects features by iteratively adding or removing features based on model performance.  

### **Syntax**  
```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
selector = SequentialFeatureSelector(model, n_features_to_select=5, direction="forward", cv=5)
X_new = selector.fit_transform(X, y)
```

### **Parameters**  
| **Parameter**  | **Description** |
|---------------|----------------|
| `direction`   | `"forward"` (adds features) or `"backward"` (removes features) |
| `cv`          | Number of cross-validation folds |
| `n_features_to_select` | Number of features to retain |

### **Use Case**  
- Works well when testing all feature combinations is computationally expensive.  
- Forward selection is faster for large feature sets.  

---

## **Choosing the Right Wrapper Method**  

| **Scenario** | **Recommended Method** |
|-------------|------------------------|
| Model supports feature importance | **RFE** |
| Need automatic feature selection | **RFECV** |
| Large feature sets with high computation cost | **SFS** |

Wrapper methods optimize feature selection for specific models, improving prediction accuracy but requiring more computational resources.