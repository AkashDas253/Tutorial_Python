## **Boolean (`bool`) in Python**  

### **Definition**  
- A Boolean represents one of two values: **`True`** or **`False`**.  
- Internally, `True` is equivalent to `1`, and `False` is equivalent to `0`.  

```python
x = True
y = False
print(x + 1)  # 2 (True is treated as 1)
print(y + 1)  # 1 (False is treated as 0)
```

---

### **Boolean Values from Other Data Types**  
| Data Type | True Values | False Values |
|-----------|------------|--------------|
| Numbers | Any nonzero number (`1, -3, 3.5`) | `0` |
| Strings | Any non-empty string (`"Hello"`) | `""` (empty string) |
| Lists, Tuples, Sets, Dicts | Any non-empty collection (`[1]`, `(0,)`, `{1: "a"}`) | Empty collections (`[]`, `{}`, `()`, `set()`) |
| `None` | - | `None` |

```python
print(bool(0))       # False
print(bool(1))       # True
print(bool(""))      # False
print(bool("Hi"))    # True
print(bool([]))      # False
print(bool([1, 2]))  # True
print(bool(None))    # False
```

---

### **Boolean Operators**  
| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `and` | Returns `True` if both are `True` | `True and False` | `False` |
| `or` | Returns `True` if at least one is `True` | `True or False` | `True` |
| `not` | Negates a Boolean value | `not True` | `False` |

```python
a = True
b = False

print(a and b)  # False
print(a or b)   # True
print(not a)    # False
```

---

### **Comparison Operators (Return Boolean Values)**
| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `==` | Equal to | `5 == 5` | `True` |
| `!=` | Not equal to | `5 != 3` | `True` |
| `>` | Greater than | `10 > 5` | `True` |
| `<` | Less than | `10 < 5` | `False` |
| `>=` | Greater than or equal | `5 >= 5` | `True` |
| `<=` | Less than or equal | `3 <= 5` | `True` |

```python
print(5 == 5)  # True
print(10 > 5)  # True
print(3 <= 2)  # False
```

---

### **Boolean in Conditional Statements**
```python
x = 10

if x > 5:
    print("Greater than 5")  # Output: Greater than 5
else:
    print("Less than or equal to 5")
```

---

### **Using `bool()` Function**
```python
print(bool(10))  # True
print(bool(0))   # False
print(bool("Python"))  # True
print(bool(""))  # False
```

---

### **Boolean Short-Circuiting**
- **`and`** stops at the first `False` value.
- **`or`** stops at the first `True` value.

```python
print(0 and 10)  # 0 (stops at False)
print(5 or 10)   # 5 (stops at True)
```

---
---

## Boolean in Python

### Introduction
In Python, the `bool` type is used to represent truth values: `True` and `False`. These values are used in conditions and control flow structures like `if` statements, loops, etc. Internally, `True` is equivalent to `1` and `False` is equivalent to `0`.

### Creating Booleans
You can create a boolean value in Python by explicitly assigning `True` or `False` to a variable.

```python
is_active = True
is_completed = False
```

### Type Checking
To check the type of a variable, you can use the `type()` function.

```python
is_active = True
print(type(is_active))  # Output: <class 'bool'>
```

### Boolean Operations
Python supports several boolean operations. The most common ones are:

| Operation        | Symbol   | Example          | Result   |
|------------------|----------|------------------|----------|
| AND              | `and`    | `True and False`  | `False`  |
| OR               | `or`     | `True or False`   | `True`   |
| NOT              | `not`    | `not True`        | `False`  |

#### Example of Boolean Operations:

```python
x = True
y = False

# AND operation
print(x and y)  # Output: False

# OR operation
print(x or y)  # Output: True

# NOT operation
print(not x)   # Output: False
```

### Conditional Statements with Booleans
Booleans are commonly used in `if` statements for controlling the flow of the program.

```python
x = True
if x:
    print("Condition is True")
else:
    print("Condition is False")
```

### Comparison Operations
Booleans are often returned from comparison operations. Python supports various comparison operators:

| Operation        | Symbol   | Example          | Result   |
|------------------|----------|------------------|----------|
| Equal to         | `==`     | `5 == 5`         | `True`   |
| Not Equal to     | `!=`     | `5 != 4`         | `True`   |
| Greater than     | `>`      | `5 > 4`          | `True`   |
| Less than        | `<`      | `5 < 4`          | `False`  |
| Greater or Equal | `>=`     | `5 >= 5`         | `True`   |
| Less or Equal    | `<=`     | `5 <= 5`         | `True`   |

#### Example of Comparison Operations:
```python
a = 10
b = 5

print(a > b)    # Output: True
print(a == b)   # Output: False
```

### Type Conversion
You can convert other types to booleans using the `bool()` function. In Python:
- `0`, `None`, `[]`, `{}`, and `''` (empty string) are considered `False`.
- Other values are considered `True`.

#### Example of Type Conversion:
```python
a = 0
print(bool(a))  # Output: False

b = "Hello"
print(bool(b))  # Output: True
```

### Built-in Functions for Boolean Type

1. **`bool(x)`**: Converts `x` to a boolean value.
   ```python
   result = bool(10)  # True
   result = bool(0)   # False
   ```

2. **`all(iterable)`**: Returns `True` if all elements in the iterable are true.
   ```python
   result = all([True, True, False])  # False
   ```

3. **`any(iterable)`**: Returns `True` if any element in the iterable is true.
   ```python
   result = any([True, False, False])  # True
   ```

### Truthy and Falsy Values
- **Truthy**: Any value that evaluates to `True` in a boolean context. Non-zero numbers, non-empty strings, and non-empty collections are truthy.
- **Falsy**: Values that evaluate to `False` in a boolean context. Common falsy values are `0`, `None`, `False`, empty sequences (e.g., `[]`, `{}`, `''`), and empty objects.

### Example of Truthy and Falsy Values:
```python
# Falsy values
print(bool(0))       # Output: False
print(bool(None))    # Output: False
print(bool(''))      # Output: False
print(bool([]))      # Output: False

# Truthy values
print(bool(1))       # Output: True
print(bool('Hello')) # Output: True
print(bool([1, 2]))  # Output: True
```

### Summary

- **Boolean Creation**: Use `True` and `False` to create booleans.
- **Boolean Operations**: Includes `and`, `or`, and `not`.
- **Comparison Operations**: Supports comparison operators like `==`, `!=`, `<`, `>`, etc.
- **Type Conversion**: Use `bool()` to convert values to booleans.
- **Truthy/Falsy**: Non-zero numbers, non-empty strings, and collections are truthy; `0`, `None`, empty strings, and collections are falsy.
- **Built-in Functions**: Includes `bool()`, `all()`, and `any()` for working with booleans.
