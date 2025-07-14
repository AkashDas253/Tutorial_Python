## **Functions in Python**  

Functions are reusable blocks of code that perform a specific task. They improve code organization, reusability, and readability.

---

## **Defining and Calling Functions**  

### **Syntax**  
```python
def function_name(parameters):
    # Function body
    return value  # Optional
```

### **Example**  
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

**Output:**  
```
Hello, Alice!
```

---

## **Function Parameters and Arguments**  

| Type | Description |
|------|------------|
| **Positional Arguments** | Passed in order |
| **Keyword Arguments** | Specified with parameter names |
| **Default Arguments** | Have predefined values |
| **Variable-length Arguments** | Accept multiple arguments (`*args`, `**kwargs`) |

### **Positional Arguments**  
```python
def add(a, b):
    return a + b

print(add(3, 5))
```
**Output:**  
`8`

---

### **Keyword Arguments**  
```python
def greet(name, msg):
    return f"{msg}, {name}!"

print(greet(name="Alice", msg="Good morning"))
```
**Output:**  
`Good morning, Alice!`

---

### **Default Arguments**  
```python
def greet(name, msg="Hello"):
    return f"{msg}, {name}!"

print(greet("Alice"))
print(greet("Bob", "Good evening"))
```
**Output:**  
```
Hello, Alice!
Good evening, Bob!
```

---

### **Variable-length Arguments**  

#### **`*args` (Multiple Positional Arguments)**
```python
def total(*numbers):
    return sum(numbers)

print(total(1, 2, 3, 4, 5))
```
**Output:**  
`15`

---

#### **`**kwargs` (Multiple Keyword Arguments)**
```python
def info(**details):
    return details

print(info(name="Alice", age=25, city="New York"))
```
**Output:**  
`{'name': 'Alice', 'age': 25, 'city': 'New York'}`

---

## **Function Scope**  

| Scope | Description |
|-------|-------------|
| **Local** | Defined inside a function |
| **Global** | Defined outside any function |
| **Nonlocal** | Modifies variables in enclosing functions |

### **Local Scope**  
```python
def example():
    x = 10  # Local variable
    return x

print(example())
# print(x)  # Error: x is not accessible outside
```

---

### **Global Scope**  
```python
x = 10  # Global variable

def example():
    global x  # Modify global variable
    x = 20

example()
print(x)
```
**Output:**  
`20`

---

### **Nonlocal Scope**  
```python
def outer():
    x = 5
    def inner():
        nonlocal x
        x = 10
    inner()
    print(x)

outer()
```
**Output:**  
`10`

---

## **Lambda (Anonymous) Functions**  

### **Syntax**  
```python
lambda arguments: expression
```

### **Example**  
```python
square = lambda x: x * x
print(square(5))
```
**Output:**  
`25`

---

## **Higher-Order Functions**  
Functions that take other functions as arguments.

### **`map()` Example**  
```python
nums = [1, 2, 3]
squared = list(map(lambda x: x ** 2, nums))
print(squared)
```
**Output:**  
`[1, 4, 9]`

---

### **`filter()` Example**  
```python
nums = [1, 2, 3, 4]
evens = list(filter(lambda x: x % 2 == 0, nums))
print(evens)
```
**Output:**  
`[2, 4]`

---

### **`reduce()` Example**  
```python
from functools import reduce
nums = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, nums)
print(product)
```
**Output:**  
`24`

---
