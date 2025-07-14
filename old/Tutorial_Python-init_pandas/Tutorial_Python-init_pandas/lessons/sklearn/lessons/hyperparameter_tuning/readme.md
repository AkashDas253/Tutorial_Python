## **Hyperparameter Tuning in Scikit-Learn**  

### **Overview**  
Hyperparameter tuning is the process of selecting the best hyperparameters for a machine learning model to optimize performance. Scikit-Learn provides several methods for tuning, including **Grid Search, Random Search, and Bayesian Optimization**.

---

### **Methods for Hyperparameter Tuning**  

| Method                | Description |
|-----------------------|-------------|
| **GridSearchCV**      | Exhaustively searches all possible hyperparameter combinations. |
| **RandomizedSearchCV** | Selects random hyperparameter combinations to find the best model efficiently. |
| **Bayesian Optimization (Optuna, Hyperopt)** | Uses probabilistic modeling to find the best hyperparameters with fewer evaluations. |
| **Manual Tuning**      | Adjusts hyperparameters manually based on model performance. |

---

### **Syntax with Parameters**  

#### **1. Grid Search Cross-Validation (GridSearchCV)**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define model
model = RandomForestClassifier()

# Define hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=model,      # Model to tune
    param_grid=param_grid, # Hyperparameter grid
    scoring='accuracy',    # Performance metric
    cv=5,                  # Number of folds
    verbose=1,             # Print progress
    n_jobs=-1              # Use all CPU cores
)

# Fit model
grid_search.fit(X, y)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
```

---

#### **2. Random Search Cross-Validation (RandomizedSearchCV)**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define hyperparameters (using distributions for random sampling)
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11)
}

# Perform Random Search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,         # Number of random combinations
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
```

---

#### **3. Bayesian Optimization (Optuna Example)**
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best Parameters:", study.best_params)
```

---

### **Choosing the Right Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| Small search space | **GridSearchCV** |
| Large search space | **RandomizedSearchCV** |
| High-dimensional tuning | **Bayesian Optimization (Optuna, Hyperopt)** |
| Custom tuning | **Manual tuning using domain knowledge** |

---