## **Cross-Validation in Scikit-Learn**  

### **Overview**  
Cross-validation is a resampling technique used to evaluate machine learning models by splitting the dataset into multiple parts and training/testing on different subsets. It helps in assessing model performance and detecting overfitting.  

---

### **Types of Cross-Validation**  

| Type                     | Description |
|--------------------------|-------------|
| **K-Fold Cross-Validation** | Splits data into *K* subsets, using *K-1* for training and 1 for testing, iterating *K* times. |
| **Stratified K-Fold** | Ensures class distribution is maintained across folds (useful for imbalanced datasets). |
| **Leave-One-Out (LOO)** | Uses a single instance for testing, with the rest for training. |
| **Leave-P-Out (LPO)** | Leaves *P* instances for testing, using the rest for training. |
| **ShuffleSplit** | Randomly splits data into train-test pairs multiple times. |
| **TimeSeriesSplit** | Used for time-dependent data, ensuring past data is never leaked into future predictions. |

---

### **Syntax with Parameters**  

#### **1. K-Fold Cross-Validation**
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

# K-Fold Cross-Validation
kf = KFold(
    n_splits=5,   # Number of splits
    shuffle=True, # Shuffle before splitting
    random_state=42
)

# Model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
```

#### **2. Stratified K-Fold Cross-Validation**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,    # Number of splits
    shuffle=True,  # Shuffle before splitting
    random_state=42
)
```

#### **3. Leave-One-Out (LOO) Cross-Validation**
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

print("Mean accuracy:", scores.mean())
```

#### **4. Leave-P-Out (LPO) Cross-Validation**
```python
from sklearn.model_selection import LeavePOut

lpo = LeavePOut(p=2)  # Leaves out 2 samples per iteration
```

#### **5. Shuffle Split Cross-Validation**
```python
from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(
    n_splits=5,    # Number of train-test splits
    test_size=0.2, # 20% test data
    random_state=42
)
```

#### **6. Time Series Split (For Sequential Data)**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

---

### **Choosing the Right Cross-Validation Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| General case | **K-Fold Cross-Validation** |
| Imbalanced dataset | **Stratified K-Fold** |
| Small dataset | **Leave-One-Out (LOO)** |
| Custom splits with a fixed test size | **ShuffleSplit** |
| Time-series data | **TimeSeriesSplit** |
| High computation cost | **K-Fold with fewer splits (e.g., K=5)** |

---