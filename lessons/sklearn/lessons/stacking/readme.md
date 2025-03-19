## **Stacking in Scikit-Learn**  

### **Overview**  
Stacking (Stacked Generalization) is an ensemble learning technique that combines multiple base models (level-0) and a meta-model (level-1) to improve predictive performance. The base models generate predictions, which are then used as features by the meta-model.  

---

### **How Stacking Works**  
1. Train multiple base models on the dataset.  
2. Collect their predictions as new features.  
3. Train a meta-model on these new features to make final predictions.  

---

### **Stacking Methods in Scikit-Learn**  

| Method | Description | Use Case |
|--------|-------------|----------|
| **StackingClassifier** | Combines multiple classifiers and a meta-classifier. | Used when individual models capture different patterns in data. |
| **StackingRegressor** | Combines multiple regressors and a meta-regressor. | Suitable when different regression models have complementary strengths. |

---

### **StackingClassifier**  

**Usage**:  
- Used when different models capture different aspects of the data.  
- Helps improve generalization compared to individual classifiers.  

**Syntax**:  
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define base models
base_models = [
    ('decision_tree', DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(probability=True)),
    ('random_forest', RandomForestClassifier(n_estimators=100))
]

# Define meta-model
meta_model = LogisticRegression()

# Initialize Stacking Classifier
model = StackingClassifier(
    estimators=base_models,  # Base learners
    final_estimator=meta_model,  # Meta-learner
    stack_method='auto'
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

---

### **StackingRegressor**  

**Usage**:  
- Used when different regressors provide diverse predictions.  
- Helps improve predictive performance compared to a single regressor.  

**Syntax**:  
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Define base models
base_models = [
    ('decision_tree', DecisionTreeRegressor(max_depth=5)),
    ('random_forest', RandomForestRegressor(n_estimators=100))
]

# Define meta-model
meta_model = Ridge()

# Initialize Stacking Regressor
model = StackingRegressor(
    estimators=base_models,  # Base learners
    final_estimator=meta_model  # Meta-learner
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

---

### **Choosing the Right Stacking Model**  

| Scenario | Recommended Method |
|----------|--------------------|
| Improve classification by combining models | **StackingClassifier** |
| Enhance regression performance using diverse models | **StackingRegressor** |
| When base models capture different patterns in data | **StackingClassifier/StackingRegressor** |

---