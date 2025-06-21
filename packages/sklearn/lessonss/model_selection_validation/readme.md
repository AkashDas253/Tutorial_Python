## 🧠 **Model Selection & Validation in Scikit-Learn**

Model selection and validation are essential for building reliable, generalizable models. Scikit-learn offers various utilities to evaluate model performance, perform cross-validation, and tune hyperparameters.

---

### 📘 **1. Cross-Validation Techniques in Scikit-Learn**
Here is the **comprehensive note on Cross-Validation Techniques** in Scikit-Learn with **complete list**, **comparison table**, and **syntax** — refined and extended from your version:

| Technique                      | Class Name                | Description                                                   | Best Use Case                                |
| ------------------------------ | ------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| **K-Fold**                     | `KFold`                   | Splits data into *K* equal-sized folds                        | Balanced datasets                            |
| **Stratified K-Fold**          | `StratifiedKFold`         | Preserves label distribution in each fold                     | Classification with class imbalance          |
| **Group K-Fold**               | `GroupKFold`              | Ensures samples from the same group aren't split across folds | Data grouped by ID (e.g., patient, customer) |
| **Leave-One-Out**              | `LeaveOneOut`             | Uses one sample for validation in each iteration              | Very small datasets                          |
| **Leave-P-Out**                | `LeavePOut`               | Uses *P* samples for validation in each iteration             | Small datasets with exhaustive evaluation    |
| **ShuffleSplit**               | `ShuffleSplit`            | Randomized train-test splits                                  | Fast repeated random splits                  |
| **Stratified ShuffleSplit**    | `StratifiedShuffleSplit`  | Combines label-stratification and shuffle                     | Repeated stratified splits                   |
| **Time Series Split**          | `TimeSeriesSplit`         | Preserves temporal order in splits                            | Time-dependent / sequential data             |
| **Predefined Split**           | `PredefinedSplit`         | Use manually defined fold indices                             | Custom fold allocation                       |
| **Repeated K-Fold**            | `RepeatedKFold`           | K-Fold repeated multiple times with different splits          | Boosts robustness in evaluation              |
| **Repeated Stratified K-Fold** | `RepeatedStratifiedKFold` | Stratified K-Fold repeated multiple times                     | Stratified repeated evaluation               |

---

### 📊 Comparison Highlights

| Feature                     | Randomization | Maintains Label Ratio | Suitable for Time-Series | Suitable for Groups |
| --------------------------- | ------------- | --------------------- | ------------------------ | ------------------- |
| `KFold`                     | Optional      | ❌                     | ❌                        | ❌                   |
| `StratifiedKFold`           | Optional      | ✅                     | ❌                        | ❌                   |
| `GroupKFold`                | ❌             | ❌                     | ❌                        | ✅                   |
| `LeaveOneOut` / `LeavePOut` | ❌             | ❌                     | ❌                        | ❌                   |
| `ShuffleSplit`              | ✅             | ❌                     | ❌                        | ❌                   |
| `StratifiedShuffleSplit`    | ✅             | ✅                     | ❌                        | ❌                   |
| `TimeSeriesSplit`           | ❌             | ❌                     | ✅                        | ❌                   |
| `RepeatedKFold`             | ✅             | ❌                     | ❌                        | ❌                   |
| `RepeatedStratifiedKFold`   | ✅             | ✅                     | ❌                        | ❌                   |
| `PredefinedSplit`           | ❌             | Depends               | Depends                  | Depends             |

---

### 🧪 Syntax Examples

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    LeaveOneOut, ShuffleSplit, TimeSeriesSplit,
    RepeatedKFold, RepeatedStratifiedKFold,
    PredefinedSplit, cross_val_score
)

# Basic K-Fold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold (for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group K-Fold
cv = GroupKFold(n_splits=5)

# Leave-One-Out
cv = LeaveOneOut()

# Shuffle Split
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Time Series Split
cv = TimeSeriesSplit(n_splits=5)

# Repeated K-Fold
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# Repeated Stratified K-Fold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# Cross-validation scoring
scores = cross_val_score(model, X, y, cv=cv)  # Average accuracy or other metric
```

---

### ⚙️ **2. Hyperparameter Tuning**

| Method                  | Description                                | Suitable For                |
| ----------------------- | ------------------------------------------ | --------------------------- |
| `GridSearchCV`          | Exhaustively searches over parameter grid. | Small search space.         |
| `RandomizedSearchCV`    | Samples random combos from grid.           | Large parameter grids.      |
| `HalvingGridSearchCV`   | Successive halving using full grid.        | Large models, limited time. |
| `HalvingRandomSearchCV` | Successive halving using random search.    | Fast + scalable tuning.     |

#### 🔧 Syntax

```python
from sklearn.model_selection import GridSearchCV

# Initialize GridSearchCV
grid = GridSearchCV(
    estimator=model,
    param_grid={'n_neighbors': [3, 5, 7]},  # Parameter grid
    cv=5,                   # 5-fold CV
    scoring='accuracy',    # Evaluation metric
    n_jobs=-1              # Use all CPUs
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

---

### 📊 **3. Validation & Learning Curves**

| Tool               | Use Case                                                  |
| ------------------ | --------------------------------------------------------- |
| `validation_curve` | Check effect of a parameter on training/validation score. |
| `learning_curve`   | Observe performance vs. dataset size.                     |

#### 📈 Syntax Example

```python
from sklearn.model_selection import validation_curve

# Get training and validation scores
train_scores, val_scores = validation_curve(
    estimator=model,
    X=X, y=y,
    param_name="max_depth",           # Parameter to vary
    param_range=[1, 2, 3, 4, 5],       # Values to test
    cv=5
)
```

---

### 🧮 **4. Model Evaluation Metrics**

| Type           | Metric                                | Function                                       |
| -------------- | ------------------------------------- | ---------------------------------------------- |
| Classification | Accuracy                              | `accuracy_score`                               |
| Classification | Precision, Recall, F1                 | `precision_score`, `recall_score`, `f1_score`  |
| Classification | Macro, Micro, Weighted Avg            | `f1_score(average='macro'/'micro'/'weighted')` |
| Classification | ROC-AUC                               | `roc_auc_score`                                |
| Classification | Log Loss                              | `log_loss`                                     |
| Classification | Confusion Matrix                      | `confusion_matrix`                             |
| Classification | Hamming Loss                          | `hamming_loss`                                 |
| Classification | Jaccard Score                         | `jaccard_score`                                |
| Classification | Matthews Corr. Coef. (MCC)            | `matthews_corrcoef`                            |
| Classification | Cohen’s Kappa                         | `cohen_kappa_score`                            |
| Classification | Zero-One Loss                         | `zero_one_loss`                                |
| Classification | Report                                | `classification_report`                        |
| Regression     | Mean Absolute Error (MAE)             | `mean_absolute_error`                          |
| Regression     | Mean Squared Error (MSE)              | `mean_squared_error`                           |
| Regression     | Root Mean Squared Error               | `mean_squared_error(squared=False)`            |
| Regression     | Mean Absolute Percentage Error (MAPE) | `mean_absolute_percentage_error`               |
| Regression     | Median Absolute Error                 | `median_absolute_error`                        |
| Regression     | Explained Variance Score              | `explained_variance_score`                     |
| Regression     | R² Score                              | `r2_score`                                     |

---

### 🧪 Syntax Example

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, jaccard_score,
    mean_squared_error, mean_absolute_error, r2_score
)

y_pred = model.predict(X_test)

# Classification Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Regression Metrics (for regression models)
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
# print("R² Score:", r2_score(y_test, y_pred))
```

---

### 🗃️ **5. Utility Summary Table**

| Utility             | Description                                 |
| ------------------- | ------------------------------------------- |
| `cross_val_score`   | Returns scores using cross-validation.      |
| `cross_validate`    | Returns train and test scores plus timings. |
| `cross_val_predict` | Returns predictions from CV.                |

#### Example

```python
from sklearn.model_selection import cross_val_predict

# Predict using CV
y_pred = cross_val_predict(model, X, y, cv=5)
```

---

### ✅ **Recommendations**

| Situation                    | Suggested Tool                        |
| ---------------------------- | ------------------------------------- |
| Need best model parameters   | `GridSearchCV` / `RandomizedSearchCV` |
| Small dataset, high variance | `LeaveOneOut`                         |
| Imbalanced classification    | `StratifiedKFold`, `f1_score`         |
| Time-dependent features      | `TimeSeriesSplit`                     |
| Limited time, large grid     | `HalvingRandomSearchCV`               |

---
