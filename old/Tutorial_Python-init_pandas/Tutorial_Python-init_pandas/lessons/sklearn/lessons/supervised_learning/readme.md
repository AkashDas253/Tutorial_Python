## **Supervised Learning in Scikit-Learn**  

### **Overview**  
Supervised learning is a type of machine learning where the model learns from labeled data. It is categorized into **classification** (predicting categories) and **regression** (predicting continuous values).  

---

### **Types of Supervised Learning**  

| Type | Description | Example Algorithms | Use Case |
|------|-------------|--------------------|----------|
| **Classification** | Predicts discrete labels (categories). | Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, Naïve Bayes | Email spam detection, medical diagnosis |
| **Regression** | Predicts continuous values. | Linear Regression, Ridge Regression, Decision Tree Regressor, SVR, Random Forest Regressor | House price prediction, stock price forecasting |

---

### **Classification in Scikit-Learn**  

**Usage**:  
- Used when output labels are categorical.  
- Evaluated using metrics like accuracy, precision, recall, and F1-score.  

**Syntax**:  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### **Regression in Scikit-Learn**  

**Usage**:  
- Used when output is a continuous numeric value.  
- Evaluated using metrics like RMSE, R², and MAE.  

**Syntax**:  
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train regressor
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Evaluate performance
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

### **Choosing the Right Supervised Model**  

| Scenario | Recommended Model |
|----------|--------------------|
| Predicting categorical labels | **Classification algorithms** |
| Predicting continuous values | **Regression algorithms** |
| Small datasets with linear relationships | **Linear Regression / Logistic Regression** |
| Complex relationships with non-linearity | **Decision Trees / Random Forest / SVM** |

---