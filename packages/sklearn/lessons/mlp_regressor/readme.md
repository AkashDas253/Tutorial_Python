## **MLPRegressor in Scikit-Learn**  

### **Overview**  
`MLPRegressor` is a multi-layer perceptron (MLP) model in Scikit-Learn used for regression tasks. It learns complex relationships between input and output variables using backpropagation and various activation functions.  

---

## **Key Features**  

| Feature | Description |
|---------|------------|
| **Multi-layer Architecture** | Supports multiple hidden layers. |
| **Non-linearity** | Uses activation functions like ReLU, tanh, sigmoid. |
| **Backpropagation** | Optimizes weights using gradient descent. |
| **Continuous Output** | Used for regression problems. |
| **Various Solvers** | `'adam'`, `'sgd'`, `'lbfgs'` for optimization. |

---

## **Common Parameters**  

| Parameter | Description |
|-----------|------------|
| `hidden_layer_sizes` | Tuple defining the number of neurons per layer. |
| `activation` | Activation function (`'relu'`, `'tanh'`, `'logistic'`). |
| `solver` | Optimization algorithm (`'adam'`, `'sgd'`, `'lbfgs'`). |
| `alpha` | L2 regularization strength. |
| `learning_rate` | `'constant'`, `'adaptive'`, `'invscaling'`. |
| `max_iter` | Maximum number of training iterations. |
| `random_state` | Ensures reproducibility. |

---

## **1. Using `MLPRegressor` for Regression**  
**Usage**: Predicts continuous target values based on input features.  

### **Syntax**  
```python
from sklearn.neural_network import MLPRegressor

# Initialize MLP Regressor
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100,50),  # Two hidden layers (100 and 50 neurons)
    activation='relu',            # ReLU activation function
    solver='adam',                # Adam optimizer
    max_iter=500,                 # Maximum training iterations
    random_state=42               # Ensures reproducibility
)

# Train the model
mlp_reg.fit(X_train, y_train)

# Make predictions
y_pred = mlp_reg.predict(X_test)
```

---

## **2. Evaluating Model Performance**  

### **Mean Squared Error (MSE)**  
```python
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

### **R² Score (Coefficient of Determination)**  
```python
from sklearn.metrics import r2_score

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")
```

---

## **3. Hyperparameter Tuning**  
Grid search can be used to find the best hyperparameters.  

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,50), (128,64,32)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'max_iter': [300, 500]
}

# Perform grid search
grid_search = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
```

---

## **Choosing the Right Parameters**  

| Scenario | Recommended Settings |
|----------|----------------------|
| Small dataset | Use fewer hidden layers (e.g., `(32,)`) |
| Large dataset | Use more hidden layers (e.g., `(128, 64, 32)`) |
| Faster convergence | Use `'adam'` solver |
| Avoid overfitting | Increase `alpha` (L2 regularization) |
| Non-linear patterns | Use `'relu'` activation |

---

## **Key Considerations**  

- **Training Time**: Increases with deeper networks.  
- **Overfitting**: Use regularization (`alpha`), dropout, or early stopping.  
- **Scaling**: Inputs should be scaled (e.g., using `StandardScaler`).  

`MLPRegressor` provides a flexible neural network-based regression approach in Scikit-Learn, making it suitable for a wide range of regression problems.

---