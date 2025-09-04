## **Exception and Exception Handling in Python**

### **Definition**  
- **Exception**: An error that occurs during program execution, disrupting the normal flow.  
- **Exception Handling**: Mechanism to handle errors gracefully without crashing the program.  

---

### **1. Types of Errors**  
| Type | Description | Example |
|------|------------|---------|
| **Syntax Error** | Invalid syntax in code | `if True print("Hello")` |
| **Runtime Error (Exception)** | Error occurs during execution | `1 / 0` (ZeroDivisionError) |
| **Logical Error** | Produces incorrect results but no error | Using `+` instead of `-` |

---

### **2. Common Built-in Exceptions**  
| Exception | Cause |
|-----------|-------|
| `ZeroDivisionError` | Division by zero (`1 / 0`) |
| `TypeError` | Unsupported operation between types (`"2" + 2`) |
| `ValueError` | Invalid value (`int("abc")`) |
| `IndexError` | Out-of-range list index (`lst[10]`) |
| `KeyError` | Accessing a non-existent dictionary key (`dict['key']`) |
| `AttributeError` | Calling an undefined attribute/method (`None.length()`) |
| `NameError` | Using an undefined variable (`print(x)`, where `x` is not defined) |
| `FileNotFoundError` | File does not exist (`open("file.txt")`) |

---

### **3. Exception Handling (`try-except`)**  
- Prevents program from crashing due to an error.  

```python
try:
    x = 1 / 0  # Causes ZeroDivisionError
except ZeroDivisionError:
    print("Cannot divide by zero.")  
```

**Output:**  
`Cannot divide by zero.`  

---

### **4. Handling Multiple Exceptions**  
- Multiple `except` blocks for different errors.  

```python
try:
    num = int(input("Enter a number: "))  # Causes ValueError if input is not a number
    result = 10 / num  # Causes ZeroDivisionError if num = 0
except ValueError:
    print("Invalid number entered.")
except ZeroDivisionError:
    print("Cannot divide by zero.")
```

---

### **5. Catching All Exceptions (`Exception`)**  
- Using `except Exception` to handle all errors.  

```python
try:
    print(10 / 0)  # Causes ZeroDivisionError
except Exception as e:
    print(f"Error: {e}")
```

---

### **6. `else` and `finally` Blocks**  
| Block | Purpose |
|-------|---------|
| `else` | Executes if `try` succeeds |
| `finally` | Executes **always**, whether exception occurs or not |

```python
try:
    x = 5 / 1  # No error
except ZeroDivisionError:
    print("Cannot divide by zero.")
else:
    print("Division successful.")  # Runs if no exception
finally:
    print("Execution completed.")  # Runs always
```

**Output:**  
```
Division successful.
Execution completed.
```

---

### **7. Raising Exceptions (`raise`)**  
- Manually raising exceptions using `raise`.  

```python
age = -5
if age < 0:
    raise ValueError("Age cannot be negative.")
```

---

### **8. Custom Exceptions**  
- Creating user-defined exceptions by inheriting `Exception` class.  

```python
class NegativeAgeError(Exception):
    pass

try:
    age = int(input("Enter age: "))
    if age < 0:
        raise NegativeAgeError("Age cannot be negative.")
except NegativeAgeError as e:
    print(f"Error: {e}")
```

---
