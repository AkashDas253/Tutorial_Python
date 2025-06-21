## **Voting in Scikit-Learn**  

### **Overview**  
Voting is an ensemble learning technique that combines predictions from multiple models to improve accuracy. It works by aggregating predictions using either hard voting (majority rule) or soft voting (weighted probabilities).  

---

### **Types of Voting**  

| Type | Description | Use Case |
|------|-------------|----------|
| **Hard Voting** | Predicts the class with the most votes from base classifiers. | Best for balanced datasets where each model has similar performance. |
| **Soft Voting** | Predicts based on the weighted average of probabilities from base classifiers. | Works well when models provide well-calibrated probability estimates. |

---

### **VotingClassifier**  

**Usage**:  
- Used when individual models have diverse strengths.  
- Hard voting is effective when all models are equally good.  
- Soft voting is better when models provide meaningful probability estimates.  

**Syntax**:  
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Define base models
model1 = LogisticRegression()
model2 = DecisionTreeClassifier(max_depth=5)
model3 = SVC(probability=True)  # Required for soft voting

# Initialize Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('svm', model3)],
    voting='soft'  # Change to 'hard' for majority voting
)

# Train model
voting_clf.fit(X_train, y_train)

# Predictions
y_pred = voting_clf.predict(X_test)
```

---

### **VotingRegressor**  

**Usage**:  
- Used when different regressors capture different data patterns.  
- Averages predictions from multiple regressors.  

**Syntax**:  
```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Define base models
model1 = Ridge()
model2 = DecisionTreeRegressor(max_depth=5)
model3 = RandomForestRegressor(n_estimators=100)

# Initialize Voting Regressor
voting_reg = VotingRegressor(
    estimators=[('ridge', model1), ('dt', model2), ('rf', model3)]
)

# Train model
voting_reg.fit(X_train, y_train)

# Predictions
y_pred = voting_reg.predict(X_test)
```

---

### **Choosing the Right Voting Model**  

| Scenario | Recommended Method |
|----------|--------------------|
| Combining classifiers with majority voting | **VotingClassifier (hard voting)** |
| Combining classifiers with weighted probabilities | **VotingClassifier (soft voting)** |
| Combining regressors for better predictions | **VotingRegressor** |

---