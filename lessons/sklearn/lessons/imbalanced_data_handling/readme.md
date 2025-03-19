## **Imbalanced Data Handling in Scikit-Learn**  

### **Overview**  
Imbalanced datasets occur when one class significantly outnumbers others, leading to biased models. Techniques to handle imbalanced data include resampling, class weighting, and algorithmic adjustments.  

---

## **Techniques for Handling Imbalanced Data**  

| Technique | Description | Best Use Case |
|-----------|------------|--------------|
| **Resampling (Oversampling & Undersampling)** | Adjusts class distribution by adding or removing samples. | Small datasets, improving class balance. |
| **SMOTE (Synthetic Minority Over-sampling Technique)** | Generates synthetic samples for the minority class. | Binary/multi-class classification with imbalance. |
| **Class Weighting** | Assigns higher weights to minority classes in model training. | Algorithms supporting class weights (e.g., Logistic Regression, SVM). |
| **Anomaly Detection** | Detects minority classes as anomalies. | Extreme imbalance scenarios (fraud detection). |
| **Threshold Moving** | Adjusts classification probability thresholds. | Probabilistic models like logistic regression. |

---

## **1. Resampling (Oversampling & Undersampling)**  
**Usage**: Adjusts class distributions by increasing (oversampling) or reducing (undersampling) data points.  

### **Syntax**  
```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Oversampling the minority class
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Undersampling the majority class
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

---

## **2. SMOTE (Synthetic Minority Over-sampling Technique)**  
**Usage**: Generates synthetic minority class samples using nearest neighbors.  

### **Syntax**  
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## **3. Class Weighting**  
**Usage**: Assigns higher importance to the minority class during training.  

### **Syntax**  
```python
from sklearn.linear_model import LogisticRegression

# Train model with class weights
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)
```

---

## **4. Anomaly Detection for Minority Classes**  
**Usage**: Detects rare classes as anomalies in extreme imbalance scenarios.  

### **Syntax**  
```python
from sklearn.ensemble import IsolationForest

# Train anomaly detection model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)
outliers = model.predict(X)
```

---

## **5. Threshold Moving**  
**Usage**: Adjusts the classification threshold to favor the minority class.  

### **Syntax**  
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Adjust threshold
probs = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Custom threshold
preds = np.where(probs > threshold, 1, 0)
```

---

## **Choosing the Right Technique**  

| Scenario | Recommended Technique |
|----------|------------------------|
| Small datasets with imbalance | **Resampling (Oversampling/Undersampling)** |
| Classifying minority classes without losing patterns | **SMOTE** |
| Algorithms supporting class weight adjustments | **Class Weighting** |
| Fraud detection or rare event detection | **Anomaly Detection** |
| Models providing probability scores | **Threshold Moving** |

Handling imbalanced data improves model fairness and ensures accurate predictions across all classes.

---