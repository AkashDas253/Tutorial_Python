# CatBoost Cheatsheet

## 1. Installing CatBoost
- pip install catboost  # Install CatBoost

## 2. Importing Libraries
- from catboost import CatBoostRegressor, CatBoostClassifier  # Import CatBoost classes
- import pandas as pd  # Import Pandas for data handling
- from sklearn.model_selection import train_test_split  # Import train_test_split

## 3. Loading Data
- df = pd.read_csv('data.csv')  # Load data from a CSV file
- X = df.drop('target', axis=1)  # Features
- y = df['target']  # Target variable

## 4. Splitting Data
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

## 5. Defining the Model
- model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)  # Define regression model
- model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6)  # Define classification model

## 6. Training the Model
- model.fit(X_train, y_train)  # Train the model

## 7. Making Predictions
- y_pred = model.predict(X_test)  # Make predictions

## 8. Evaluating the Model
- from sklearn.metrics import mean_squared_error  # Import metrics
- mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
- print('MSE:', mse)  # Print MSE

## 9. Feature Importance
- import matplotlib.pyplot as plt  # Import Matplotlib
- feature_importances = model.get_feature_importance()  # Get feature importances
- plt.barh(range(len(feature_importances)), feature_importances)  # Plot feature importance
- plt.show()  # Show plot

## 10. Hyperparameter Tuning
- from sklearn.model_selection import GridSearchCV  # Import GridSearchCV
- grid = GridSearchCV(estimator=CatBoostRegressor(), param_grid={
  - 'depth': [4, 6, 8],
  - 'learning_rate': [0.01, 0.1]
}, scoring='neg_mean_squared_error', cv=3)  # Define grid search
- grid.fit(X_train, y_train)  # Fit grid search
- print('Best parameters:', grid.best_params_)  # Print best parameters

## 11. Saving and Loading Model
- model.save_model('model.cbm')  # Save model
- loaded_model = CatBoostRegressor()  # Create model object
- loaded_model.load_model('model.cbm')  # Load model

## 12. Cross-validation
- from catboost import cv  # Import cross-validation
- cv_data = cv(
  - model.get_params(),  # Model parameters
  pool=lgb.Dataset(X_train, label=y_train),  # Training dataset
  fold_count=5,  # Number of folds
  shuffle=True,  # Shuffle data
  plot=True  # Plot results
)  # Perform cross-validation
- print(cv_data)  # Print cross-validation results
