## **Calibration Curves in Scikit-Learn**  

### **Overview**  
Calibration curves assess how well a classifier’s predicted probabilities align with actual outcomes. A well-calibrated model should have predicted probabilities that match observed probabilities, forming a diagonal line in the calibration plot.  

---

## **How Calibration Curves Work**  
1. **Divide predicted probabilities** into bins (e.g., [0, 0.1], [0.1, 0.2], etc.).  
2. **Compute observed frequency** of positive class in each bin.  
3. **Plot observed vs. predicted probabilities** to evaluate calibration.  

---

## **Syntax for Calibration Curves in Scikit-Learn**  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Compute calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

# Plot calibration curve
plt.plot(prob_pred, prob_true, marker='o', label="Model Calibration")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")  # Reference line
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Probability")
plt.title("Calibration Curve")
plt.legend()
plt.show()
```

---

## **Interpreting Calibration Curves**  
- **Perfect calibration**: The curve follows the diagonal line (`y=x`).  
- **Overconfident model**: The curve lies below the diagonal (predicted probabilities are too high).  
- **Underconfident model**: The curve lies above the diagonal (predicted probabilities are too low).  

---

## **Use Cases**  
- Evaluating **probability reliability** in classifiers.  
- Adjusting predictions in **imbalanced datasets**.  
- Improving decision-making in **risk-sensitive applications** (e.g., medical diagnosis, fraud detection).  

Calibration curves help assess whether a model’s predicted probabilities reflect true likelihoods, ensuring better trust and interpretability in machine learning predictions.

---