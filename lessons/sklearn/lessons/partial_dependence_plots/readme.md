## **Partial Dependence Plots (PDP) in Scikit-Learn**  

### **Overview**  
Partial Dependence Plots (PDP) show how a specific feature affects the model’s predictions by marginalizing over other features. They help understand the global relationship between a feature and the target variable.  

---

## **How PDP Works**  
1. Select one or more features.  
2. Vary the selected feature while keeping other features constant.  
3. Compute the average prediction over all instances.  
4. Plot the results to visualize how the feature influences the model.  

---

## **Syntax for PDP in Scikit-Learn**  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute partial dependence for a single feature
features = [0]  # Index or column name of the feature
pdp = partial_dependence(model, X_train, features=features)

# Plot PDP
fig, ax = plt.subplots()
ax.plot(pdp['values'][0], pdp['average'][0], marker='o')
ax.set_xlabel('Feature Value')
ax.set_ylabel('Predicted Output')
ax.set_title('Partial Dependence Plot')
plt.show()
```

---

## **Interpreting PDP**  
- **Upward trend**: Feature positively influences the target.  
- **Downward trend**: Feature negatively affects the target.  
- **Flat curve**: Feature has little to no impact.  

---

## **Use Cases**  
- Explains **global feature importance** in any model.  
- Helps understand **non-linear relationships** between features and predictions.  
- Works with **both classification and regression** models.  

PDP provides valuable insights into model behavior, improving interpretability and trust in machine learning predictions.

---