## **Built-in Datasets in Scikit-Learn**  

### **Overview**  
Scikit-Learn provides several built-in datasets that are useful for testing machine learning models. These datasets are small, load quickly, and are available through `load_*` functions.  

---

## **1. Common Built-in Datasets**  

| Dataset | Function | Description | Task Type |
|---------|----------|-------------|------------|
| **Iris** | `load_iris()` | Flower dataset with 3 species | Classification |
| **Digits** | `load_digits()` | Handwritten digits (0-9) images | Classification |
| **Wine** | `load_wine()` | Chemical analysis of wine samples | Classification |
| **Breast Cancer** | `load_breast_cancer()` | Diagnostic data for breast cancer | Classification |
| **Diabetes** | `load_diabetes()` | Continuous diabetes progression data | Regression |

---

## **2. Loading Built-in Datasets**  

### **Syntax**  
```python
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Extract features and target
X, y = iris.data, iris.target

# Display dataset information
print("Feature names:", iris.feature_names)  # Column names
print("Target names:", iris.target_names)    # Class labels
print("Shape of X:", X.shape)                # Data shape
```

---

## **3. Understanding Dataset Components**  

| Attribute | Description |
|-----------|-------------|
| `.data` | Feature matrix (2D NumPy array) |
| `.target` | Target values (1D NumPy array) |
| `.feature_names` | List of feature names |
| `.target_names` | List of class labels |
| `.DESCR` | Dataset description |

#### **Example: Exploring the Digits Dataset**  
```python
# Load the Digits dataset
digits = datasets.load_digits()

# Print dataset description
print(digits.DESCR)
```

---

## **4. Choosing the Right Dataset**  

| Scenario | Recommended Dataset |
|----------|----------------------|
| **Classification (Multiclass)** | `load_iris()`, `load_wine()`, `load_digits()` |
| **Binary Classification** | `load_breast_cancer()` |
| **Regression** | `load_diabetes()` |

---