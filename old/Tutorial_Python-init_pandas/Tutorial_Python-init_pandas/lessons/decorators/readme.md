## **Decorators in Python**  

A **decorator** is a function that modifies the behavior of another function **without changing its code**. It is used to add functionality dynamically.

---

## **1. Basic Syntax of a Decorator**  
```python
def decorator(func):
    def wrapper():
        print("Before function execution")
        func()
        print("After function execution")
    return wrapper
```

---

## **2. Applying a Decorator**  

### **Using `@decorator_name`**
```python
def my_decorator(func):
    def wrapper():
        print("Function is about to run.")
        func()
        print("Function has run.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```
**Output:**  
```
Function is about to run.
Hello!
Function has run.
```

---

## **3. Decorating a Function with Arguments**
```python
def repeat_decorator(func):
    def wrapper(*args, **kwargs):
        print("Repeating function call:")
        return func(*args, **kwargs)
    return wrapper

@repeat_decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```
**Output:**  
```
Repeating function call:
Hello, Alice!
```

---

## **4. Returning a Value from Decorated Function**
```python
def double_result(func):
    def wrapper(x):
        return func(x) * 2
    return wrapper

@double_result
def square(n):
    return n * n

print(square(3))  # Output: 18 (instead of 9)
```

---

## **5. Decorating Functions with `functools.wraps`**
Using `functools.wraps` preserves the original function name and docstring.
```python
from functools import wraps

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args}")
        return func(*args, **kwargs)
    return wrapper

@log_function
def add(a, b):
    """Returns sum of two numbers."""
    return a + b

print(add(2, 3))  # Output: Calling add with (2, 3) → 5
```

---

## **6. Chaining Multiple Decorators**
```python
def bold(func):
    def wrapper():
        return f"<b>{func()}</b>"
    return wrapper

def italic(func):
    def wrapper():
        return f"<i>{func()}</i>"
    return wrapper

@bold
@italic
def text():
    return "Hello"

print(text())  # Output: <b><i>Hello</i></b>
```

---

## **7. Class-Based Decorators**
```python
class Logger:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"Executing {self.func.__name__}")
        return self.func(*args, **kwargs)

@Logger
def multiply(a, b):
    return a * b

print(multiply(2, 3))  # Output: Executing multiply → 6
```

---
