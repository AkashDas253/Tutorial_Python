## **Flow Control in Python**

Flow control determines how statements execute in a program. It includes:  
- **Conditional statements** (`if`, `elif`, `else`)  
- **Looping constructs** (`for`, `while`)  
- **Control statements** (`break`, `continue`, `pass`, `return`, `yield`, `exit()`, `quit()`, `sys.exit()`)

---

## **Conditional Statements**  

### **Syntax**
```python
if condition:
    # Executes if condition is True
elif condition:
    # Executes if previous conditions are False
else:
    # Executes if all conditions are False
```

### **Example**
```python
x = 10
if x > 10:
    print("Greater")
elif x == 10:
    print("Equal")
else:
    print("Smaller")
```

**Output:**  
`Equal`

---

## **Loops**  

### **`for` Loop**  

### **Syntax**
```python
for variable in sequence:
    # Loop body
```

### **Example**
```python
for i in range(3):
    print(i)
```

**Output:**  
```
0
1
2
```

---

### **`while` Loop**  

### **Syntax**
```python
while condition:
    # Loop body
```

### **Example**
```python
x = 0
while x < 3:
    print(x)
    x += 1
```

**Output:**  
```
0
1
2
```

---

## **Loop Control Statements**  

| Statement | Purpose | Syntax |
|-----------|---------|--------|
| `break` | Exits loop immediately | `break` |
| `continue` | Skips current iteration | `continue` |
| `pass` | Placeholder, does nothing | `pass` |

### **Examples**  

#### **`break` Example**
```python
for i in range(5):
    if i == 3:
        break
    print(i)
```
**Output:**  
```
0
1
2
```

---

#### **`continue` Example**
```python
for i in range(5):
    if i == 3:
        continue
    print(i)
```
**Output:**  
```
0
1
2
4
```

---

#### **`pass` Example**
```python
for i in range(3):
    if i == 1:
        pass  # Placeholder
    print(i)
```
**Output:**  
```
0
1
2
```

---

## **Function Control Statements**  

| Statement | Purpose | Syntax |
|-----------|---------|--------|
| `return` | Exits function and returns a value | `return value` |
| `yield` | Returns generator value and continues execution | `yield value` |
| `pass` | Placeholder, does nothing | `pass` |

### **`return` Example**  
```python
def add(a, b):
    return a + b

print(add(5, 3))
```
**Output:**  
`8`

---

### **`yield` Example**  
```python
def generator():
    for i in range(3):
        yield i

gen = generator()
print(next(gen))
print(next(gen))
```
**Output:**  
```
0
1
```

---

## **Exit Control Statements**  

| Statement | Purpose | Syntax |
|-----------|---------|--------|
| `exit()` | Terminates program | `exit()` |
| `quit()` | Terminates program | `quit()` |
| `sys.exit()` | Terminates program with exit code | `sys.exit(code)` |

### **Example**
```python
import sys
print("Before exit")
sys.exit()
print("After exit")  # This line will not execute
```

---
