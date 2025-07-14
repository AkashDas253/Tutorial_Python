## **Embedded Methods in Feature Selection**  

### **Overview**  
Embedded methods integrate feature selection within the model training process, balancing efficiency and accuracy. They leverage the model’s feature importance scores to select relevant features automatically.  

---

## **1. LASSO (L1 Regularization) for Feature Selection**  
**Usage**: LASSO (Least Absolute Shrinkage and Selection Operator) assigns zero weights to less important features, effectively removing them.  

### **Syntax**  
```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

model = Lasso(alpha=0.01)  # L1 regularization strength
selector = SelectFromModel(model)
X_new = selector.fit_transform(X, y)
```

### **Use Case**  
- Suitable for high-dimensional data.  
- Works best for linear models.  

---

## **2. Ridge Regression (L2 Regularization) for Feature Selection**  
**Usage**: While Ridge regression does not eliminate features, it reduces their impact, helping with multicollinearity.  

### **Syntax**  
```python
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel

model = Ridge(alpha=1.0)  # L2 regularization strength
selector = SelectFromModel(model, threshold="median")
X_new = selector.fit_transform(X, y)
```

### **Use Case**  
- Reduces feature importance rather than eliminating features.  
- Best for models affected by multicollinearity.  

---

## **3. Decision Trees and Random Forest Feature Importance**  
**Usage**: Tree-based models provide feature importance scores that can be used for selection.  

### **Syntax**  
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

model = RandomForestClassifier(n_estimators=100)
selector = SelectFromModel(model, threshold="mean")
X_new = selector.fit_transform(X, y)
```

### **Use Case**  
- Effective for non-linear relationships.  
- Handles high-dimensional datasets well.  

---

## **Choosing the Right Embedded Method**  

| **Scenario** | **Recommended Method** |
|-------------|------------------------|
| Sparse features, need automatic selection | **LASSO (L1 Regularization)** |
| Multicollinearity present | **Ridge Regression (L2 Regularization)** |
| Feature importance-based selection | **Random Forest / Decision Trees** |

Embedded methods are computationally efficient while maintaining high predictive performance.