## **Local Variables in Python**

---

### **Overview**

* A **local variable** is declared **inside a function or method**.
* It exists **only within the block** in which it is defined.
* **Destroyed automatically** once the function exits.
* **Not accessible outside** its function or method scope.

---

### **Scope & Lifetime**

| Aspect   | Description                                     |
| -------- | ----------------------------------------------- |
| Scope    | Limited to the function or method               |
| Lifetime | Exists only during function/method execution    |
| Storage  | In the call stack frame of the current function |

---

### **General Example**

```python
def greet():
    msg = "Hello"  # Local variable
    print(msg)

greet()
# print(msg) → NameError: msg is not defined
```

---

### **Local Variables in OOP**

```python
class Sample:
    def display(self):
        temp = 123  # Local variable
        print(temp)

obj = Sample()
obj.display()
# print(temp) → NameError
```

* Local variables like `temp` above **cannot be used outside** the method.
* Do **not** use `self.` with local variables.

---

### **Difference from Instance/Class/Global Variables**

| Type              | Declared In                | Access Scope         | Uses `self`/`cls` | Shared? |
| ----------------- | -------------------------- | -------------------- | ----------------- | ------- |
| Local Variable    | Inside a function/method   | That function/method | ❌ No              | ❌ No    |
| Instance Variable | Inside method using `self` | Whole object         | ✅ Yes             | ❌ No    |
| Class Variable    | Inside class (not method)  | Whole class          | ✅ Yes             | ✅ Yes   |
| Global Variable   | Outside all classes        | Whole module/file    | ❌ No              | ✅ Yes   |

---

### **Shadowing Global Variables**

```python
x = 10

def example():
    x = 5  # Local variable shadows global `x`
    print(x)

example()     # 5
print(x)      # 10 (global remains unchanged)
```

---

### **Modifying Global from Inside Function**

```python
x = 10

def modify():
    global x
    x = 20

modify()
print(x)  # 20
```

---

### **Best Practices**

* Use **local variables** for **temporary, method-specific values**.
* Avoid using the same name as a global/class variable to reduce confusion.

---

## **Local Variables in Python OOP**

---

### **Overview**

* **Local variables** are variables **declared inside a method/function**.
* They exist **only during the execution** of that method.
* Not accessible outside the method/block where defined.

---

### **Scope**

* Their scope is **limited to the function or method**.
* Automatically destroyed once the function call ends.

---

### **Declaration Example**

```python
class Example:
    def show(self):
        message = "Hello!"  # Local variable
        print(message)

obj = Example()
obj.show()
# print(message) → NameError (message is not accessible here)
```

---

### **Characteristics**

| Feature             | Local Variable          |
| ------------------- | ----------------------- |
| Defined inside      | Method or function      |
| Lifetime            | During method execution |
| Accessed via        | Only inside the method  |
| Uses `self`?        | ❌ No                    |
| Visible to instance | ❌ No                    |

---

### **Difference from Instance & Class Variables**

| Type              | Declared In           | Scope                   | Access Modifier |
| ----------------- | --------------------- | ----------------------- | --------------- |
| Local Variable    | Inside method         | That method only        | N/A             |
| Instance Variable | `self.var` in methods | Whole object lifecycle  | Yes             |
| Class Variable    | Class body            | Shared across instances | Yes             |

---

### **Usage Example**

```python
class Calculator:
    def add(self, a, b):
        result = a + b  # Local variable
        return result

calc = Calculator()
print(calc.add(5, 3))  # 8
# print(result) → NameError
```

---

### **Key Point**

* Local variables are useful for **temporary computations** or **intermediate values** that don’t need to be stored in the object.

---
