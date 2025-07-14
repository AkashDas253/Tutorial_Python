## **Method Overloading in Python**

---

### **Overview**

* **Method Overloading** allows defining **multiple methods with the same name** but **different parameters**.
* Unlike some languages (e.g., Java or C++), **Python does not support true method overloading** by default.
* Python handles method definitions **dynamically**, and **only the last defined method is used** if names are repeated.

---

### **Python's Behavior**

* Python methods can take a **variable number of arguments** using default arguments, `*args`, and `**kwargs`.
* True overloading based on different signatures is **not supported**—last method definition overrides earlier ones.

---

### **Example: Not Supported Natively**

```python
class Demo:
    def show(self, x):
        print("One argument", x)

    def show(self, x, y):
        print("Two arguments", x, y)

d = Demo()
d.show(1, 2)   # Works: Two arguments 1 2
# d.show(1)    # Error: missing 1 required positional argument
```

* `show(self, x)` is **overridden** by `show(self, x, y)`.

---

### **Workaround Using Default Arguments**

```python
class Demo:
    def show(self, x=None, y=None):
        if x is not None and y is not None:
            print("Two arguments", x, y)
        elif x is not None:
            print("One argument", x)
        else:
            print("No arguments")

d = Demo()
d.show(1, 2)  # Output: Two arguments 1 2
d.show(1)     # Output: One argument 1
d.show()      # Output: No arguments
```

---

### **Workaround Using `*args` and `**kwargs`**

```python
class Demo:
    def show(self, *args):
        if len(args) == 1:
            print("One argument", args[0])
        elif len(args) == 2:
            print("Two arguments", args[0], args[1])
        else:
            print("Unsupported number of arguments")

d = Demo()
d.show(10)
d.show(10, 20)
```

---

### **Using `@singledispatchmethod` (Python 3.8+)**

* The `functools.singledispatchmethod` decorator can simulate overloading **based on the type of the first argument**.

```python
from functools import singledispatchmethod

class Demo:
    @singledispatchmethod
    def show(self, arg):
        print("Default:", arg)

    @show.register
    def _(self, arg: int):
        print("Integer:", arg)

    @show.register
    def _(self, arg: str):
        print("String:", arg)

d = Demo()
d.show(100)     # Integer: 100
d.show("abc")   # String: abc
```

---

### **Summary**

| Aspect                  | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| True Overloading        | ❌ Not supported natively                               |
| Default Arguments       | ✅ Simulate multiple behaviors                          |
| `*args`/`**kwargs`      | ✅ Handle variable arguments                            |
| `@singledispatchmethod` | ✅ Type-based dispatch for first argument (Python 3.8+) |
| Behavior                | Only last method with same name is retained            |

---
