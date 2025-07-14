## **Ensemble Methods in Scikit-Learn**  

### **Overview**  
Ensemble methods combine multiple weak learners to improve predictive performance. Scikit-Learn provides various ensemble techniques, including **bagging, boosting, and stacking**.

---

### **Types of Ensemble Methods**  

| Method          | Description |
|---------------|-------------|
| **Bagging**   | Uses multiple base learners trained independently on random subsets of data (e.g., Random Forest). |
| **Boosting**  | Sequentially trains weak models, each correcting previous errors (e.g., AdaBoost, Gradient Boosting, XGBoost). |
| **Stacking**  | Combines multiple models and trains a meta-model on their predictions. |
| **Voting**    | Aggregates predictions from different models using majority vote (classification) or averaging (regression). |

---

### **Bagging: Random Forest**  

**Usage**:  
- Used when reducing overfitting in high-variance models.  
- Effective for structured/tabular data classification and regression tasks.  

**Syntax**:  
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
model = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    max_depth=None,     # Maximum depth of each tree
    min_samples_split=2, # Minimum samples to split a node
    n_jobs=-1,         # Use all CPU cores
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

---

### **Boosting: AdaBoost**  

**Usage**:  
- Suitable for imbalanced data and cases requiring better generalization.  
- Works well with decision trees and weak learners.  

**Syntax**:  
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize AdaBoost with Decision Tree
model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1), # Weak learner
    n_estimators=50,     # Number of weak classifiers
    learning_rate=1.0,   # Step size
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

---

### **Boosting: Gradient Boosting**  

**Usage**:  
- Used for structured data where feature importance is critical.  
- Handles missing values well and reduces bias.  

**Syntax**:  
```python
from sklearn.ensemble import GradientBoostingClassifier

# Initialize Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

---

### **Voting Classifier**  

**Usage**:  
- Used when combining different models to improve performance.  
- Works well in multi-class classification problems.  

**Syntax**:  
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define base models
log_clf = LogisticRegression()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier(n_estimators=100)

# Initialize Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('svm', svm_clf), ('rf', rf_clf)],
    voting='hard'  # Use 'soft' for probability-based averaging
)

# Train model
voting_clf.fit(X_train, y_train)

# Predictions
y_pred = voting_clf.predict(X_test)
```

---

### **Stacking**  

**Usage**:  
- Used when combining multiple diverse models and training a meta-model.  
- Effective for tabular datasets with complex relationships.  

**Syntax**:  
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('nb', GaussianNB())
]

# Meta-model
meta_model = LogisticRegression()

# Initialize Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# Train model
stacking_clf.fit(X_train, y_train)

# Predictions
y_pred = stacking_clf.predict(X_test)
```

---

### **Choosing the Right Ensemble Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| Reduce overfitting | **Bagging (Random Forest, Extra Trees)** |
| Improve weak learners | **Boosting (AdaBoost, Gradient Boosting, XGBoost)** |
| Combine diverse models | **Stacking** |
| Use multiple models for voting | **Voting Classifier** |

---