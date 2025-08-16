# Overloading in Python

In programming, **overloading** means defining multiple behaviors for the same operator or function name, depending on the context (arguments, types, or usage).

In Python, overloading is supported in two main forms:

* **Operator Overloading** → redefining built-in operators for user-defined classes.
* **Function/Method Overloading** → multiple definitions of the same function with different arguments (limited in Python, achieved differently).

---

## Operator Overloading

Python allows operators (`+`, `-`, `*`, etc.) to be **overloaded** by defining special (dunder) methods in classes.

### Common Overloadable Operators

| Operator | Method                      |
| -------- | --------------------------- |
| `+`      | `__add__(self, other)`      |
| `-`      | `__sub__(self, other)`      |
| `*`      | `__mul__(self, other)`      |
| `/`      | `__truediv__(self, other)`  |
| `//`     | `__floordiv__(self, other)` |
| `%`      | `__mod__(self, other)`      |
| `**`     | `__pow__(self, other)`      |
| `==`     | `__eq__(self, other)`       |
| `!=`     | `__ne__(self, other)`       |
| `<`      | `__lt__(self, other)`       |
| `<=`     | `__le__(self, other)`       |
| `>`      | `__gt__(self, other)`       |
| `>=`     | `__ge__(self, other)`       |
| `[]`     | `__getitem__(self, key)`    |
| `in`     | `__contains__(self, item)`  |
| `str()`  | `__str__(self)`             |
| `len()`  | `__len__(self)`             |

### Syntax Example

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __add__(self, other):  # Overload +
        return Vector(self.x + other.x, self.y + other.y)
    
    def __str__(self):  # Overload str()
        return f"({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)  # (4, 6)
```

---

## Function / Method Overloading

Unlike languages such as Java or C++, **Python does not natively support function overloading** (same name, different signatures).
The **last defined function** with the same name overrides the previous ones.

### Ways to Achieve Function Overloading

#### 1. Default Arguments

```python
def greet(name="User", message="Hello"):
    print(f"{message}, {name}")

greet()                   # Hello, User
greet("Alice")            # Hello, Alice
greet("Alice", "Hi")      # Hi, Alice
```

#### 2. Variable-Length Arguments

```python
def add(*args):
    return sum(args)

print(add(1, 2))        # 3
print(add(1, 2, 3, 4))  # 10
```

#### 3. Type Checking (Manual Overloading)

```python
def area(x, y=None):
    if y is None:  # Circle
        return 3.14 * x * x
    else:          # Rectangle
        return x * y

print(area(5))      # Circle area
print(area(5, 10))  # Rectangle area
```

#### 4. `functools.singledispatch` (Generic Function Overloading)

```python
from functools import singledispatch

@singledispatch
def process(value):
    print("Default:", value)

@process.register(int)
def _(value):
    print("Integer:", value)

@process.register(str)
def _(value):
    print("String:", value)

process(10)      # Integer: 10
process("Hi")    # String: Hi
process([1, 2])  # Default: [1, 2]
```

---

## Method Overloading in Classes

Python methods can simulate overloading using **default args, `*args`, or type checks**.

```python
class Math:
    def product(self, a, b=1, c=1):
        return a * b * c

m = Math()
print(m.product(2))       # 2
print(m.product(2, 3))    # 6
print(m.product(2, 3, 4)) # 24
```

---

## Method Overriding vs Overloading

| Aspect         | Overloading                                                       | Overriding                                               |
| -------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| Definition     | Same function/operator behaves differently depending on arguments | Subclass provides new implementation for parent’s method |
| Where          | Within same class (operators/functions)                           | Across class hierarchy                                   |
| Python Support | Limited (via defaults, `*args`, `singledispatch`)                 | Fully supported                                          |

---

## Summary

* **Operator Overloading** → Implemented using special methods (`__add__`, `__str__`, etc.).
* **Function Overloading** → Not natively supported; achieved via **default args, `*args`, type checks, or `singledispatch`**.
* **Method Overloading** → Same as function overloading but inside classes.
* **Overloading vs Overriding** → Overloading = multiple forms in same class, Overriding = redefined in subclass.

---
