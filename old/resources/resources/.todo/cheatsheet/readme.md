# Python Cheatsheet

## 1. Installing Python
- Download from python.org  # Install Python

## 2. Basic Syntax
- print("Hello, World!")  # Print to console
- x = 5  # Variable assignment
- y = x + 10  # Arithmetic operation

## 3. Data Types
- int_var = 10  # Integer
- float_var = 10.5  # Float
- str_var = "Hello"  # String
- list_var = [1, 2, 3]  # List
- tuple_var = (1, 2, 3)  # Tuple
- dict_var = {'key': 'value'}  # Dictionary
- set_var = {1, 2, 3}  # Set

## 4. Control Structures
- if x > 5:
  - print("x is greater than 5")  # If statement
- for i in range(5):
  - print(i)  # For loop
- while x > 0:
  - x -= 1  # While loop

## 5. Functions
- def my_function(param1, param2):
  - return param1 + param2  # Function definition

- result = my_function(5, 3)  # Function call

- def my_function(*args):
  - return sum(args)  # Function definition with *args

- result = my_function(1, 2, 3, 4)  # Function call with *args

- def my_function(**kwargs):
  - for key, value in kwargs.items():
    - print(f"{key} = {value}")  # Function definition with **kwargs

- my_function(arg1=1, arg2=2, arg3=3)  # Function call with **kwargs

## 6. Importing Libraries
- import math  # Import standard library
- from datetime import datetime  # Import specific module

## 7. File Handling
- with open('file.txt', 'r') as file:  # Open file for reading
  - content = file.read()  # Read file content
- with open('file.txt', 'w') as file:  # Open file for writing
  - file.write("Hello, World!")  # Write to file

## 8. Exception Handling
- try:
  - result = 10 / 0  # Code that may raise an exception
- except ZeroDivisionError:
  - print("Division by zero error!")  # Handle exception

## 9. List Comprehensions
- squares = [x**2 for x in range(10)]  # Create a list of squares

## 10. Lambda Functions
- add = lambda x, y: x + y  # Define a lambda function
- result = add(5, 3)  # Call the lambda function

## 11. Classes and Objects
- class MyClass:
  - def __init__(self, value):
    - self.value = value  # Initialize instance variable

  - def display(self):
    - print(self.value)  # Instance method

- obj = MyClass(10)  # Create an object
- obj.display()  # Call method

## 12. List Methods
- list_var.append(4)  # Add element
- list_var.remove(2)  # Remove element
- length = len(list_var)  # Get length
- sorted_list = sorted(list_var)  # Sort list

## 13. Dictionary Methods
- dict_var['new_key'] = 'new_value'  # Add key-value pair
- value = dict_var.get('key')  # Get value by key
- keys = dict_var.keys()  # Get all keys
- values = dict_var.values()  # Get all values

## 14. Set Methods
- set_var.add(4)  # Add element
- set_var.remove(2)  # Remove element
- intersection = set_var.intersection({1, 2, 3, 4})  # Set intersection

## 15. String Methods
- str_var.lower()  # Convert to lowercase
- str_var.upper()  # Convert to uppercase
- str_var.split(',')  # Split string into a list
- str_var.replace('old', 'new')  # Replace substring

## 16. NumPy Basics
- import numpy as np  # Import NumPy
- np.array([1, 2, 3])  # Create a NumPy array
- np.zeros((2, 3))  # Create a 2D array of zeros
- np.ones((3, 2))  # Create a 2D array of ones
- np.arange(0, 10, 2)  # Create an array with a range of numbers
- np.random.rand(3, 2)  # Create a 2D array with random numbers

## 17. Pandas Basics
- import pandas as pd  # Import Pandas
- pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})  # Create a DataFrame
- df.head()  # View the first few rows of DataFrame
- df.describe()  # Get summary statistics
- df['col1']  # Access a column

## 18. Regular Expressions
- import re  # Import re module
- re.search(r'pattern', 'string')  # Search for a pattern
- re.match(r'pattern', 'string')  # Match a pattern at the start
- re.sub(r'pattern', 'replacement', 'string')  # Substitute pattern in string

## 19. List and Dictionary Comprehensions
- squares = [x**2 for x in range(10)]  # List comprehension
- even_dict = {x: x**2 for x in range(5) if x % 2 == 0}  # Dictionary comprehension

## 20. Environment Management
- python -m venv myenv  # Create a virtual environment
- source myenv/bin/activate  # Activate virtual environment (Linux/Mac)
- myenv\Scripts\activate  # Activate virtual environment (Windows)

## 21. Commonly Used Libraries
- import requests  # For making HTTP requests
- import matplotlib.pyplot as plt  # For plotting
- import seaborn as sns  # For statistical data visualization
- import scikit-learn as skl  # For machine learning

## 22. Logging
- import logging  # Import logging module
- logging.basicConfig(level=logging.INFO)  # Set logging level
- logging.info('This is an info message')  # Log an info message

## 23. Working with Dates and Times
- from datetime import datetime  # Import datetime
- now = datetime.now()  # Get current date and time
- date_str = now.strftime("%Y-%m-%d")  # Format date

## 24. List Slicing
- list_var[0:3]  # Get elements from index 0 to 2
- list_var[-1]  # Get the last element
- list_var[::2]  # Get every second element

## 25. Multi-threading
- import threading  # Import threading module
- def thread_function():
  - print("Thread executing")  # Thread function

- thread = threading.Thread(target=thread_function)  # Create thread
- thread.start()  # Start thread
- thread.join()  # Wait for thread to finish

## 26. AsyncIO
- import asyncio  # Import asyncio module
- async def async_function():
  - await asyncio.sleep(1)  # Asynchronous sleep

- asyncio.run(async_function())  # Run the async function

## 27. Comprehensions for Sets
- unique_squares = {x**2 for x in range(10)}  # Set comprehension

## 28. Property Decorators
- class MyClass:
  - def __init__(self, value):
    - self._value = value  # Private variable

  - @property
  - def value(self):
    - return self._value  # Getter

  - @value.setter
  - def value(self, new_value):
    - self._value = new_value  # Setter

## 29. Context Managers
- with open('file.txt', 'r') as file:  # Context manager
  - content = file.read()  # Read file

## 30. Decorators
- def my_decorator(func):  # Create a decorator
  - def wrapper():
    - print("Something is happening before the function is called.")
    - func()
    - print("Something is happening after the function is called.")
  - return wrapper

- @my_decorator  # Apply the decorator
- def say_hello():
  - print("Hello!")

## 31. Generators
- def my_generator():
  - yield 1  # Yield a value
  - yield 2  # Yield another value

- gen = my_generator()  # Create a generator object
- next(gen)  # Get the next value from the generator

## 32. Iterators
- class MyIterator:
  - def __init__(self):
    - self.current = 0  # Initialize the iterator

  - def __iter__(self):
    - return self  # Return the iterator object

  - def __next__(self):
    - if self.current < 5:
      - self.current += 1
      - return self.current  # Return the next value
    - raise StopIteration  # Stop iteration

## 33. File Formats
- import json  # For JSON handling
- import csv  # For CSV handling

- with open('data.json', 'r') as f:  # Read JSON
  - data = json.load(f)

- with open('data.csv', 'w', newline='') as f:  # Write CSV
  - writer = csv.writer(f)
  - writer.writerow(['Name', 'Age'])  # Write header

## 34. Contextlib
- from contextlib import contextmanager  # Import context manager

- @contextmanager
- def my_context():
  - print("Entering the context")
  - yield  # Yield control to the context
  - print("Exiting the context")

- with my_context():  # Use the context manager
  - print("Inside the context")

## 35. Handling JSON
- import json  # Import JSON module
- data = {'key': 'value'}  # Create a dictionary
- json_string = json.dumps(data)  # Convert to JSON string
- loaded_data = json.loads(json_string)  # Load JSON string back to dictionary

## 36. SQLAlchemy
- from sqlalchemy import create_engine  # Import SQLAlchemy
- engine = create_engine('sqlite:///mydatabase.db')  # Create database engine

## 37. Unit Testing
- import unittest  # Import unit testing framework

- class TestMyFunction(unittest.TestCase):
  - def test_add(self):
    - self.assertEqual(add(2, 3), 5)  # Test case

- if __name__ == '__main__':
  - unittest.main()  # Run the tests

## 38. String Formatting
- f"Hello, {name}!"  # f-string formatting
- "Hello, {}".format(name)  # str.format() method

## 39. CSV Handling
- import csv  # Import CSV module

- with open('file.csv', 'r') as file:  # Open CSV file
  - reader = csv.reader(file)  # Create CSV reader
  - for row in reader:
    - print(row)  # Print each row

## 40. Handling Command Line Arguments
- import sys  # Import sys module

- script_name = sys.argv[0]  # Get script name
- arguments = sys.argv[1:]  # Get command line arguments

## 41. Type Hinting
- def add(x: int, y: int) -> int:  # Function with type hints
  - return x + y  # Return sum

## 42. Assertions
- assert x > 0, "x must be positive"  # Assert statement

## 43. Multi-processing
- from multiprocessing import Process  # Import Process

- def worker_function():
  - print("Worker executing")  # Worker function

- process = Process(target=worker_function)  # Create process
- process.start()  # Start process
- process.join()  # Wait for process to finish

## 44. Pretty Printing
- import pprint  # Import pprint module

- data = {'a': [1, 2, 3], 'b': {'c': 4}}  # Complex data structure
- pprint.pprint(data)  # Pretty print data

## 45. Environment Variables
- import os  # Import os module

- os.environ['MY_ENV_VAR'] = 'value'  # Set environment variable
- value = os.getenv('MY_ENV_VAR')  # Get environment variable