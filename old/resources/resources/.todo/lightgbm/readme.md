# LightGBM Cheatsheet

## 1. Installing LightGBM
- pip install lightgbm  # Install LightGBM

## 2. Importing Libraries
- import lightgbm as lgb  # Import LightGBM library
- import pandas as pd  # Import Pandas for data handling
- from sklearn.model_selection import train_test_split  # Import train_test_split

## 3. Loading Data
- df = pd.read_csv('data.csv')  # Load data from a CSV file
- X = df.drop('target', axis=1)  # Features
- y = df['target']  # Target variable

## 4. Splitting Data
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

## 5. Creating Datasets
- dtrain = lgb.Dataset(X_train, label=y_train)  # Create training dataset
- dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)  # Create test dataset

## 6. Setting Hyperparameters
- params = {
  - 'objective': 'regression',  # Objective function
  - 'metric': 'rmse',  # Evaluation metric
  - 'boosting_type': 'gbdt',  # Boosting type
  - 'num_leaves': 31,  # Maximum number of leaves
  - 'learning_rate': 0.05,  # Learning rate
  - 'feature_fraction': 0.9  # Fraction of features to consider
}  # Define hyperparameters

## 7. Training the Model
- model = lgb.train(params, dtrain, num_boost_round=100)  # Train the model

## 8. Making Predictions
- y_pred = model.predict(X_test, num_iteration=model.best_iteration)  # Make predictions

## 9. Evaluating the Model
- from sklearn.metrics import mean_squared_error  # Import metrics
- mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
- print('MSE:', mse)  # Print MSE

## 10. Feature Importance
- import matplotlib.pyplot as plt  # Import Matplotlib
- lgb.plot_importance(model)  # Plot feature importance
- plt.show()  # Show plot

## 11. Hyperparameter Tuning
- from sklearn.model_selection import GridSearchCV  # Import GridSearchCV
- grid = GridSearchCV(estimator=lgb.LGBMRegressor(), param_grid={
  - 'num_leaves': [31, 50],
  - 'learning_rate': [0.01, 0.1]
}, scoring='neg_mean_squared_error', cv=3)  # Define grid search
- grid.fit(X_train, y_train)  # Fit grid search
- print('Best parameters:', grid.best_params_)  # Print best parameters

## 12. Saving and Loading Model
- model.save_model('model.txt')  # Save model
- loaded_model = lgb.Booster(model_file='model.txt')  # Load model

## 13. Cross-validation
- cv_results = lgb.cv(params, dtrain, num_boost_round=100, nfold=5, metrics='rmse', as_pandas=True)  # Perform cross-validation
- print(cv_results)  # Print cross-validation results
