## **Parallelization in Scikit-Learn**  

### **Overview**  
Parallelization in Scikit-Learn helps speed up model training, evaluation, and processing by utilizing multiple CPU cores. This is achieved using the `n_jobs` parameter, efficient libraries like `joblib`, and parallel computation techniques.

---

## **1. `n_jobs` Parameter for Parallel Processing**  
Most Scikit-Learn estimators support parallelization using `n_jobs`, which controls the number of CPU cores used.

| `n_jobs` Value | Description |
|---------------|-------------|
| `n_jobs=-1`  | Uses all available CPU cores. |
| `n_jobs=1`   | Runs sequentially (default). |
| `n_jobs=N`   | Uses `N` CPU cores. |

### **Syntax: Parallel Training in Random Forest**  
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest with parallel processing
clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    n_jobs=-1,         # Use all CPU cores
    random_state=42    # Ensures reproducibility
)
clf.fit(X_train, y_train)
```

---

## **2. Parallel Model Selection & Hyperparameter Tuning**  
Hyperparameter tuning can be parallelized using `n_jobs`.

### **Syntax: Grid Search with Parallel Processing**  
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define hyperparameter grid
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Initialize Grid Search with parallelization
grid_search = GridSearchCV(
    SVC(),           # Model
    param_grid,      # Parameter grid
    cv=5,            # 5-fold cross-validation
    n_jobs=-1        # Use all CPU cores
)
grid_search.fit(X_train, y_train)
```

---

## **3. Joblib for Explicit Parallelization**  
`joblib` is used for parallel computation, especially for tasks that involve repeated function calls.

### **Syntax: Parallel Computing with `joblib.Parallel`**  
```python
from joblib import Parallel, delayed
import time

# Dummy function
def process(i):
    time.sleep(1)
    return i * i

# Parallel execution using all CPU cores
results = Parallel(n_jobs=-1)(delayed(process)(i) for i in range(10))
print(results)
```

---

## **4. Parallel Feature Selection**  
Feature selection methods in Scikit-Learn also support parallel processing.

### **Syntax: Parallel Feature Selection with `RFECV`**  
```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Initialize Recursive Feature Elimination with parallelization
selector = RFECV(
    RandomForestClassifier(),  # Model
    step=1,                    # Remove one feature at a time
    cv=5,                      # Cross-validation folds
    n_jobs=-1                  # Use all CPU cores
)
selector.fit(X_train, y_train)
```

---

## **5. Parallel Training in Ensemble Methods**  
Ensemble methods like `BaggingClassifier`, `GradientBoostingClassifier`, and `RandomForestClassifier` allow parallel training.

### **Syntax: Bagging Classifier with Parallelization**  
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize Bagging Classifier with parallel execution
model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),  # Weak learner
    n_estimators=50,       # Number of classifiers
    max_samples=0.8,       # Fraction of data for each model
    max_features=0.8,      # Fraction of features used
    bootstrap=True,        # Sample with replacement
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)
model.fit(X_train, y_train)
```

---

## **6. GPU Acceleration for Scikit-Learn (CuML)**  
Scikit-Learn does not natively support GPUs, but CuML provides GPU-accelerated implementations.

### **Example: Using CuML’s Random Forest**  
```python
from cuml.ensemble import RandomForestClassifier

# Initialize GPU-based Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

---

## **7. When to Use Parallelization?**  

| Scenario | Parallelization Method |
|----------|------------------------|
| **Training large models** | Set `n_jobs=-1` in estimators. |
| **Hyperparameter tuning** | Use `GridSearchCV(n_jobs=-1)`. |
| **Feature selection** | Use `RFECV(n_jobs=-1)`. |
| **Custom function execution** | Use `joblib.Parallel`. |
| **Ensemble methods** | Use `n_jobs=-1` in `RandomForestClassifier`, `BaggingClassifier`. |

---