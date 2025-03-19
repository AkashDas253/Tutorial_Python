## **Regression Models in Scikit-Learn**  

### **Overview**  
Regression models predict a continuous numerical value based on input features. Scikit-Learn provides various regression algorithms, each suitable for different types of relationships in data.  

---

### **Types of Regression Models**  

| Model | Description | Best Use Case |
|-------|-------------|--------------|
| **Linear Regression** | Fits a straight line to the data. | Simple linear relationships. |
| **Ridge Regression** | Adds L2 regularization to Linear Regression. | Prevents overfitting in high-dimensional data. |
| **Lasso Regression** | Adds L1 regularization, leading to feature selection. | Sparse models where some features are irrelevant. |
| **Polynomial Regression** | Extends Linear Regression with polynomial terms. | Capturing non-linear relationships. |
| **Support Vector Regression (SVR)** | Uses Support Vector Machines for regression. | Handles complex, high-dimensional data. |
| **Decision Tree Regressor** | Uses a tree structure to model data. | Non-linear relationships, interpretability. |
| **Random Forest Regressor** | Uses multiple decision trees (bagging). | Reduces overfitting compared to a single tree. |
| **Gradient Boosting Regressor** | Uses boosting to improve predictions iteratively. | High-performance regression with feature interactions. |

---

### **Regression Model Implementations**  

#### **1. Linear Regression**  
**Usage**: Used for data with a linear relationship between input and output.  
**Syntax**:  
```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression(
    fit_intercept=True,  # Whether to calculate the intercept
    copy_X=True,         # Whether to copy X
    n_jobs=None,         # Number of jobs to run in parallel
    positive=False       # Restrict coefficients to be positive
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **2. Ridge Regression**  
**Usage**: Used when multicollinearity is present. Helps in regularization by penalizing large coefficients.  
**Syntax**:  
```python
from sklearn.linear_model import Ridge

reg = Ridge(
    alpha=1.0,         # Regularization strength
    fit_intercept=True # Whether to calculate the intercept
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **3. Lasso Regression**  
**Usage**: Performs feature selection by reducing some coefficients to zero.  
**Syntax**:  
```python
from sklearn.linear_model import Lasso

reg = Lasso(
    alpha=0.1,         # Regularization strength
    fit_intercept=True # Whether to calculate the intercept
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **4. Polynomial Regression**  
**Usage**: Used when the relationship between variables is non-linear.  
**Syntax**:  
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

reg = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True), # Polynomial transformation
    LinearRegression()
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **5. Support Vector Regression (SVR)**  
**Usage**: Works well with small datasets and non-linear relationships.  
**Syntax**:  
```python
from sklearn.svm import SVR

reg = SVR(
    kernel='rbf',  # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,         # Regularization parameter
    epsilon=0.1    # Margin of tolerance
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **6. Decision Tree Regressor**  
**Usage**: Handles complex relationships but is prone to overfitting.  
**Syntax**:  
```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    max_depth=5,       # Maximum depth of the tree
    min_samples_split=2 # Minimum samples required to split a node
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **7. Random Forest Regressor**  
**Usage**: Reduces overfitting by combining multiple decision trees.  
**Syntax**:  
```python
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of trees
    random_state=42    # Random seed for reproducibility
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

#### **8. Gradient Boosting Regressor**  
**Usage**: Boosting technique that improves weak learners iteratively.  
**Syntax**:  
```python
from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(
    n_estimators=100,  # Number of boosting stages
    learning_rate=0.1, # Rate at which the model learns
    max_depth=3        # Maximum depth of each tree
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

### **Choosing the Right Regression Model**  

| Scenario | Recommended Model |
|----------|--------------------|
| Data has a linear relationship | **Linear Regression** |
| High-dimensional data with multicollinearity | **Ridge Regression** |
| Feature selection required | **Lasso Regression** |
| Non-linear relationships present | **Polynomial Regression / Decision Tree Regressor** |
| Small dataset with complex patterns | **SVR** |
| Need better generalization than Decision Trees | **Random Forest Regressor** |
| Need highly optimized regression performance | **Gradient Boosting Regressor** |

---