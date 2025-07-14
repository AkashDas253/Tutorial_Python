# XGBoost Cheatsheet

## 1. Installing XGBoost
- pip install xgboost  # Install XGBoost

## 2. Importing Libraries
- import xgboost as xgb  # Import XGBoost library
- import pandas as pd  # Import Pandas for data handling
- from sklearn.model_selection import train_test_split  # Import train_test_split

## 3. Loading Data
- df = pd.read_csv('data.csv')  # Load data from a CSV file
- X = df.drop('target', axis=1)  # Features
- y = df['target']  # Target variable

## 4. Splitting Data
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

## 5. Creating DMatrix
- dtrain = xgb.DMatrix(X_train, label=y_train)  # Create DMatrix for training
- dtest = xgb.DMatrix(X_test, label=y_test)  # Create DMatrix for testing

## 6. Setting Hyperparameters
- params = {
  - 'objective': 'reg:squarederror',  # Objective function
  - 'max_depth': 3,  # Maximum tree depth
  - 'eta': 0.1,  # Learning rate
  - 'eval_metric': 'rmse'  # Evaluation metric
}  # Define hyperparameters

## 7. Training the Model
- model = xgb.train(params, dtrain, num_boost_round=100)  # Train the model

## 8. Making Predictions
- y_pred = model.predict(dtest)  # Make predictions

## 9. Evaluating the Model
- from sklearn.metrics import mean_squared_error  # Import metrics
- mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
- print('MSE:', mse)  # Print MSE

## 10. Feature Importance
- import matplotlib.pyplot as plt  # Import Matplotlib
- xgb.plot_importance(model)  # Plot feature importance
- plt.show()  # Show plot

## 11. Hyperparameter Tuning
- from sklearn.model_selection import GridSearchCV  # Import GridSearchCV
- grid = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid={
  - 'max_depth': [3, 4, 5],
  - 'eta': [0.01, 0.1, 0.2]
}, scoring='neg_mean_squared_error', cv=3)  # Define grid search
- grid.fit(X_train, y_train)  # Fit grid search
- print('Best parameters:', grid.best_params_)  # Print best parameters

## 12. Saving and Loading Model
- model.save_model('model.json')  # Save model
- loaded_model = xgb.Booster()  # Create Booster object
- loaded_model.load_model('model.json')  # Load model

## 13. Cross-validation
- cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=5, metrics='rmse', as_pandas=True)  # Perform cross-validation
- print(cv_results)  # Print cross-validation results
