# Ray Cheatsheet

## 1. Installing Ray
- pip install ray  # Install Ray

## 2. Importing Libraries
- import ray  # Import Ray

## 3. Initializing Ray
- ray.init()  # Initialize Ray

## 4. Defining Remote Functions
- @ray.remote  # Decorator to define a remote function
- def remote_function(x):  # Define the function
  - return x * x  # Function logic

## 5. Calling Remote Functions
- result = remote_function.remote(5)  # Call the remote function
- print(ray.get(result))  # Get the result

## 6. Defining Remote Classes
- @ray.remote  # Decorator to define a remote class
- class RemoteActor:
  - def __init__(self, value):
    - self.value = value  # Initialize actor state
  
  - def get_value(self):
    - return self.value  # Get the actor's value

## 7. Creating and Using Actors
- actor = RemoteActor.remote(10)  # Create an actor instance
- value = actor.get_value.remote()  # Call actor method
- print(ray.get(value))  # Get the result

## 8. Parallel Execution
- futures = [remote_function.remote(i) for i in range(10)]  # Launch multiple tasks
- results = ray.get(futures)  # Get results from all tasks
- print(results)  # Print results

## 9. Using Ray with DataFrames
- import pandas as pd  # Import Pandas
- df = pd.DataFrame({'x': range(10)})  # Create a DataFrame

- @ray.remote  # Define a remote function for DataFrame operations
- def compute_square(df):
  - return df['x'] ** 2  # Compute square of the column

- future = compute_square.remote(df)  # Call remote function
- result_df = ray.get(future)  # Get the result

## 10. Scaling with Ray
- from ray import tune  # Import Ray Tune for hyperparameter tuning

- def train_model(config):  # Define a training function
  - # Model training logic here
  - return {'loss': loss}  # Return the loss
  
- analysis = tune.run(train_model, config={'param': tune.grid_search([1, 2, 3])})  # Run hyperparameter tuning

## 11. Using Ray with Machine Learning
- from ray import train  # Import Ray Train
- trainer = train.Trainer(backend="torch")  # Initialize a trainer

- @trainer.run
- def train_fn():
  - # Model training logic
