## **Classification Models in Scikit-Learn**  

### **Overview**  
Classification models predict categorical labels based on input features. Scikit-Learn provides various classification algorithms, each suitable for different data distributions and complexities.  

---

### **Types of Classification Models**  

| Model | Description | Best Use Case |
|-------|-------------|--------------|
| **Logistic Regression** | Linear model for binary classification. | Binary classification problems. |
| **K-Nearest Neighbors (KNN)** | Assigns labels based on closest neighbors. | Small datasets, non-linear decision boundaries. |
| **Support Vector Machine (SVM)** | Finds the optimal hyperplane for classification. | High-dimensional and small datasets. |
| **Decision Tree Classifier** | Uses a tree-like structure for decisions. | Easy to interpret, handles non-linearity. |
| **Random Forest Classifier** | Uses multiple decision trees (bagging). | Reduces overfitting compared to a single tree. |
| **Gradient Boosting Classifier** | Uses boosting to improve predictions iteratively. | High-performance classification. |
| **Naïve Bayes Classifier** | Based on Bayes' theorem, assumes feature independence. | Text classification, spam detection. |
| **Neural Networks (MLPClassifier)** | Multi-layer perceptron for classification. | Complex patterns and large datasets. |

---

### **Classification Model Implementations**  

#### **1. Logistic Regression**  
**Usage**: Used for binary and multi-class classification problems.  
**Syntax**:  
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    penalty='l2',         # Regularization: 'l1', 'l2', 'elasticnet', or None
    C=1.0,                # Inverse of regularization strength
    solver='lbfgs',       # Optimization algorithm
    multi_class='auto'    # 'ovr' or 'multinomial' for multi-class
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **2. K-Nearest Neighbors (KNN)**  
**Usage**: Assigns the class of the majority among the k-nearest neighbors.  
**Syntax**:  
```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(
    n_neighbors=5,      # Number of neighbors
    metric='minkowski', # Distance metric
    weights='uniform'   # 'uniform' or 'distance' weighting
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **3. Support Vector Machine (SVM)**  
**Usage**: Finds a hyperplane that best separates classes.  
**Syntax**:  
```python
from sklearn.svm import SVC

clf = SVC(
    kernel='rbf',   # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,          # Regularization parameter
    gamma='scale'   # Kernel coefficient
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **4. Decision Tree Classifier**  
**Usage**: Uses a tree structure to classify instances.  
**Syntax**:  
```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',  # 'gini' or 'entropy'
    max_depth=None,    # Maximum depth of the tree
    min_samples_split=2 # Minimum samples required to split a node
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **5. Random Forest Classifier**  
**Usage**: Reduces overfitting by combining multiple decision trees.  
**Syntax**:  
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of trees
    random_state=42    # Random seed for reproducibility
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **6. Gradient Boosting Classifier**  
**Usage**: Boosting technique that improves weak learners iteratively.  
**Syntax**:  
```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(
    n_estimators=100,  # Number of boosting stages
    learning_rate=0.1, # Rate at which the model learns
    max_depth=3        # Maximum depth of each tree
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **7. Naïve Bayes Classifier**  
**Usage**: Based on Bayes' theorem, best for text classification and spam detection.  
**Syntax**:  
```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

#### **8. Multi-Layer Perceptron (Neural Network)**  
**Usage**: A simple feedforward artificial neural network for classification.  
**Syntax**:  
```python
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(
    hidden_layer_sizes=(100,), # Number of neurons in each hidden layer
    activation='relu',         # Activation function
    solver='adam',             # Optimization algorithm
    max_iter=200               # Maximum training iterations
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

### **Choosing the Right Classification Model**  

| Scenario | Recommended Model |
|----------|--------------------|
| Binary classification with linear separation | **Logistic Regression** |
| Small dataset with non-linear boundaries | **KNN** |
| High-dimensional data | **SVM** |
| Simple, interpretable model | **Decision Tree Classifier** |
| Preventing overfitting in decision trees | **Random Forest Classifier** |
| Optimized performance with boosting | **Gradient Boosting Classifier** |
| Text classification problems | **Naïve Bayes Classifier** |
| Complex patterns and large datasets | **Neural Networks (MLPClassifier)** |

---