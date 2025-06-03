## **Multiple Dispatch in Python**

---

### **Overview**

* **Multiple Dispatch** is the ability to **select a function or method to invoke based on the types of two or more arguments**.
* Unlike **single dispatch** (which depends on only the first argument’s type), multiple dispatch allows **polymorphism across several input types**.
* Python does not natively support multiple dispatch, but it can be achieved using external libraries like `multipledispatch`.

---

### **Library: `multipledispatch`**

* Install using:

  ```bash
  pip install multipledispatch
  ```

* Provided by `multipledispatch.Dispatcher` or through the `@dispatch` decorator.

---

### **Example Using `@dispatch`**

```python
from multipledispatch import dispatch

@dispatch(int, int)
def add(a, b):
    print("Adding integers:", a + b)

@dispatch(float, float)
def add(a, b):
    print("Adding floats:", a + b)

@dispatch(str, str)
def add(a, b):
    print("Concatenating strings:", a + b)

add(2, 3)         # Adding integers: 5
add(2.5, 3.1)     # Adding floats: 5.6
add("Hi", "Bye")  # Concatenating strings: HiBye
```

---

### **Why Multiple Dispatch is Useful**

* Promotes **clean and type-safe** function definitions.
* Avoids manual type-checking inside functions using `isinstance()` or `type()`.
* Useful in applications involving **mathematical operations, symbolic computation, simulations**, etc.

---

### **Difference from Single Dispatch**

| Feature             | Single Dispatch                     | Multiple Dispatch                        |
| ------------------- | ----------------------------------- | ---------------------------------------- |
| Dispatch Based On   | Type of the **first argument only** | Type of **all arguments**                |
| Built-in Support    | ✅ `functools.singledispatch`        | ❌ External (`multipledispatch`) required |
| Customization Level | Moderate                            | High                                     |

---

### **Comparison Example**

#### **Single Dispatch**

```python
from functools import singledispatch

@singledispatch
def process(arg):
    print("Default")

@process.register
def _(arg: int):
    print("Integer")

process(1)       # Integer
process("abc")   # Default
```

#### **Multiple Dispatch**

```python
from multipledispatch import dispatch

@dispatch(int, str)
def process(a, b):
    print("Int + Str")

@dispatch(str, str)
def process(a, b):
    print("Str + Str")

process(1, "x")   # Int + Str
process("a", "b") # Str + Str
```

---

### **Limitations of `multipledispatch`**

* No built-in support for **method dispatch in classes** (use `@staticmethod` workaround).
* Runtime-based; not as performant as native method overloading.
* Type annotations are required at runtime for resolution.

---

### **Summary**

| Concept           | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| Multiple Dispatch | Selects function by types of **all** arguments                          |
| Library Used      | `multipledispatch`                                                      |
| Decorator         | `@dispatch(type1, type2, ...)`                                          |
| Native Support    | ❌ Python needs external library                                         |
| Use Cases         | Mathematical ops, symbolic computing, API routing, custom type handling |

---
