## **Resampling Methods in Scikit-Learn**  

### **Overview**  
Resampling techniques help address class imbalance by modifying the dataset to improve model performance. Methods include oversampling, undersampling, and hybrid techniques.

---

## **Types of Resampling Methods**  

| Method | Description | Best Use Case |
|--------|------------|--------------|
| **Random Oversampling** | Duplicates minority class samples. | Small datasets with class imbalance. |
| **Random Undersampling** | Removes majority class samples. | Large datasets with significant imbalance. |
| **SMOTE (Synthetic Minority Over-sampling Technique)** | Generates synthetic samples using nearest neighbors. | Maintaining feature distribution in minority class. |
| **ADASYN (Adaptive Synthetic Sampling)** | Focuses on generating samples for harder-to-learn minority class examples. | When minority class has complex distributions. |
| **Tomek Links** | Removes closely paired samples from different classes. | Cleaning noisy boundaries between classes. |
| **NearMiss** | Selects majority samples close to the minority class. | When majority class is too dominant. |
| **SMOTEENN (SMOTE + Edited Nearest Neighbors)** | Applies SMOTE to generate samples, then ENN removes noisy samples. | Balancing classes while reducing noise. |

---

## **1. Random Oversampling**  
**Usage**: Duplicates minority class samples to balance class distribution.  

### **Syntax**  
```python
from imblearn.over_sampling import RandomOverSampler

# Initialize RandomOverSampler
oversampler = RandomOverSampler(
    sampling_strategy='auto',  # Determines resampling ratio
    random_state=42            # Ensures reproducibility
)
X_resampled, y_resampled = oversampler.fit_resample(X, y)
```

---

## **2. Random Undersampling**  
**Usage**: Removes random samples from the majority class.  

### **Syntax**  
```python
from imblearn.under_sampling import RandomUnderSampler

# Initialize RandomUnderSampler
undersampler = RandomUnderSampler(
    sampling_strategy='auto',  # Determines resampling ratio
    random_state=42            # Ensures reproducibility
)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

---

## **3. SMOTE (Synthetic Minority Over-sampling Technique)**  
**Usage**: Generates synthetic samples for the minority class using nearest neighbors.  

### **Syntax**  
```python
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Determines resampling ratio
    k_neighbors=5,             # Number of nearest neighbors used
    random_state=42            # Ensures reproducibility
)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## **4. ADASYN (Adaptive Synthetic Sampling)**  
**Usage**: Generates synthetic samples focusing on difficult-to-learn examples.  

### **Syntax**  
```python
from imblearn.over_sampling import ADASYN

# Initialize ADASYN
adasyn = ADASYN(
    sampling_strategy='auto',  # Determines resampling ratio
    n_neighbors=5,             # Number of nearest neighbors used
    random_state=42            # Ensures reproducibility
)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

---

## **5. Tomek Links**  
**Usage**: Removes closely paired majority-minority class samples to clean decision boundaries.  

### **Syntax**  
```python
from imblearn.under_sampling import TomekLinks

# Initialize TomekLinks
tomek = TomekLinks(
    sampling_strategy='auto'  # Removes majority class samples forming Tomek links
)
X_resampled, y_resampled = tomek.fit_resample(X, y)
```

---

## **6. NearMiss**  
**Usage**: Selects majority samples close to the minority class to balance dataset.  

### **Syntax**  
```python
from imblearn.under_sampling import NearMiss

# Initialize NearMiss
nearmiss = NearMiss(
    version=1,         # Strategy for selecting samples (1, 2, or 3)
    n_neighbors=3      # Number of neighbors considered
)
X_resampled, y_resampled = nearmiss.fit_resample(X, y)
```

---

## **7. SMOTEENN (SMOTE + Edited Nearest Neighbors)**  
**Usage**: Balances the dataset using SMOTE and then removes noisy samples with ENN.  

### **Syntax**  
```python
from imblearn.combine import SMOTEENN

# Initialize SMOTEENN
smoteenn = SMOTEENN(
    sampling_strategy='auto',  # Determines resampling ratio
    random_state=42            # Ensures reproducibility
)
X_resampled, y_resampled = smoteenn.fit_resample(X, y)
```

---

## **Choosing the Right Resampling Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| Small dataset with class imbalance | **Random Oversampling** |
| Large dataset with excessive majority samples | **Random Undersampling** |
| Need to create synthetic data without duplication | **SMOTE** |
| More complex minority class boundaries | **ADASYN** |
| Cleaning decision boundaries | **Tomek Links** |
| Majority samples close to minority class | **NearMiss** |
| Balancing data while reducing noise | **SMOTEENN** |

Resampling techniques improve model performance by ensuring a balanced class distribution for better learning.

---