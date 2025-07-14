## **Global Variables in Python**

---

### **Overview**

* **Global variables** are declared **outside of functions and classes**, at the module level.
* Accessible **throughout the module**, including inside classes and functions.
* Used to store **data shared by multiple parts** of the program.
* Modifying global variables inside functions requires the `global` keyword.

---

### **Declaring and Accessing**

```python
count = 0  # Global variable

def increment():
    global count
    count += 1

class Counter:
    def show(self):
        print(count)  # Access global variable

increment()
c = Counter()
c.show()  # Output: 1
```

---

### **Modifying Global Variables**

* To **modify** a global variable inside a function/method, use `global` keyword.
* Without `global`, assignment creates a **new local variable** instead.

```python
value = 10

def change():
    global value
    value = 20

change()
print(value)  # 20
```

---

### **Global vs Local Variable Shadowing**

```python
x = 5  # global

def func():
    x = 10  # local variable shadows global
    print(x)

func()  # 10
print(x)  # 5 (global unchanged)
```

---

### **Global Keyword Usage Inside Class Methods**

```python
total = 100

class Accumulator:
    def add(self, amount):
        global total
        total += amount

acc = Accumulator()
acc.add(50)
print(total)  # 150
```

---

### **Characteristics**

| Feature           | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| Declared at       | Module (global) scope                                           |
| Scope             | Entire module/file                                              |
| Access in methods | Directly accessible                                             |
| Modify in methods | Use `global` keyword                                            |
| Typical use cases | Constants, config, counters shared across functions and classes |

---

## **Global Variables in Python OOP**

---

### **Overview**

* **Global variables** are defined **outside of all functions and classes**.
* They are **accessible throughout the module** unless shadowed by a local variable.
* In OOP, global variables are typically used for **constants or shared data** outside class scope.

---

### **Declaration Example**

```python
x = 10  # Global variable

class Demo:
    def display(self):
        print("x =", x)

obj = Demo()
obj.display()  # x = 10
```

---

### **Accessing Global Variables in a Method**

```python
value = 100  # Global

class Test:
    def show(self):
        print("Value is", value)

t = Test()
t.show()  # Value is 100
```

---

### **Modifying Global Variables**

To modify a global variable inside a function or method, use the `global` keyword:

```python
count = 0  # Global

def increment():
    global count
    count += 1

increment()
print(count)  # 1
```

---

### **Global Keyword Inside Class Method**

```python
total = 50

class Tracker:
    def update(self):
        global total
        total += 10

t = Tracker()
t.update()
print(total)  # 60
```

---

### **Key Characteristics**

| Feature               | Global Variable                    |
| --------------------- | ---------------------------------- |
| Defined in            | Global/module scope                |
| Scope                 | Entire module                      |
| Keyword for modify    | `global`                           |
| Accessed inside class | ✅ Yes                              |
| Common use            | Constants, shared config, counters |

---

### **Difference from Other Variables**

| Variable Type | Declared In                   | Scope             | Uses `self` or `cls`    | Shared |
| ------------- | ----------------------------- | ----------------- | ----------------------- | ------ |
| Global        | Outside classes/functions     | Entire module     | ❌ No                    | ✅ Yes  |
| Class         | Inside class (not in methods) | Class & instances | ✅ via `cls` or instance | ✅ Yes  |
| Instance      | Inside methods using `self`   | Instance only     | ✅ via `self`            | ❌ No   |
| Local         | Inside method                 | That method only  | ❌ No                    | ❌ No   |

---
