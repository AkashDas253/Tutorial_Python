## **Magic Methods in Python**

---

### **Overview**

* Magic methods (also called **dunder methods**) are special methods with names surrounded by double underscores (`__method__`).
* They enable **custom behavior** of Python objects for built-in operations like arithmetic, comparison, attribute access, and more.
* Implementing magic methods allows **operator overloading** and integration with Python’s syntax.

---

### **Common Magic Methods**

| Magic Method                                                      | Purpose                                      | Example Use Case    |
| ----------------------------------------------------------------- | -------------------------------------------- | ------------------- |
| `__init__(self, ...)`                                             | Constructor, initializes new objects         | `obj = ClassName()` |
| `__str__(self)`                                                   | Informal string representation               | `print(obj)`        |
| `__repr__(self)`                                                  | Official string representation               | `repr(obj)`         |
| `__add__(self, other)`                                            | Overloads `+` operator                       | `obj1 + obj2`       |
| `__sub__(self, other)`                                            | Overloads `-` operator                       | `obj1 - obj2`       |
| `__mul__(self, other)`                                            | Overloads `*` operator                       | `obj1 * obj2`       |
| `__eq__(self, other)`                                             | Overloads equality `==`                      | `obj1 == obj2`      |
| `__lt__(self, other)`                                             | Overloads less than `<`                      | `obj1 < obj2`       |
| `__len__(self)`                                                   | Returns length                               | `len(obj)`          |
| `__getitem__(self, key)`                                          | Access elements by key/index                 | `obj[key]`          |
| `__setitem__(self, key, value)`                                   | Set elements by key/index                    | `obj[key] = value`  |
| `__delitem__(self, key)`                                          | Delete element by key/index                  | `del obj[key]`      |
| `__call__(self, ...)`                                             | Makes an object callable                     | `obj()`             |
| `__enter__(self)` and `__exit__(self, exc_type, exc_val, exc_tb)` | Context manager support for `with` statement | `with obj:`         |

---

### **Example**

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # Calls __add__
print(v3)     # Calls __str__, Output: Vector(4, 6)
```

---

### **Benefits**

* Customize how objects behave with built-in operators.
* Enable intuitive syntax for custom classes.
* Make classes integrate seamlessly with Python language features.

---
