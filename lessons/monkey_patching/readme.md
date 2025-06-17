## Monkey Patching in Python

### Overview

**Monkey patching** refers to the dynamic modification or extension of classes or modules at **runtime**. It allows you to change behavior without altering the original source code.

---

### Key Concepts

* **Dynamic Typing**: Python’s dynamic typing and reflection allow modification of objects, classes, and modules at runtime.
* **Use Case**: Commonly used in testing, fixing third-party libraries, or temporary workarounds.
* **Risks**: Can lead to hard-to-debug issues, lack of maintainability, and unpredictable behavior.

---

### Syntax and Example

```python
# Original class
class A:
    def greet(self):
        return "Hello"

# Monkey patching greet method
def new_greet(self):
    return "Hi there!"

A.greet = new_greet

obj = A()
print(obj.greet())  # Output: Hi there!
```

---

### Use Cases

| Use Case    | Description                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| Testing     | Replace functions with mocks or stubs to simulate behavior.                  |
| Bug Fixing  | Override broken behavior in third-party modules without altering the source. |
| Extension   | Add new methods to built-in or third-party classes at runtime.               |
| Prototyping | Try alternative implementations without permanent changes.                   |

---

### Monkey Patching Modules

```python
import math

# Original math.sqrt
print(math.sqrt(4))  # 2.0

# Monkey patch
math.sqrt = lambda x: "patched!"
print(math.sqrt(4))  # "patched!"
```

---

### Monkey Patching Built-in Types (Not Recommended)

```python
# Dangerous example
def fake_len(obj):
    return 42

list.__len__ = fake_len  # This will raise a TypeError: can't set attributes of built-in/extension type 'list'
```

> Built-in types in C extensions like `list`, `dict`, etc., **cannot be monkey patched directly**.

---

### Safer Alternatives

| Technique                | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| Dependency Injection     | Injecting dependencies rather than patching them.     |
| Wrappers/Decorators      | Modify behavior without altering the original object. |
| Subclassing              | Extend or override behavior via inheritance.          |
| Patch in `unittest.mock` | Temporarily patch methods during tests.               |

```python
from unittest.mock import patch

with patch('module.function_name', return_value=42):
    # Temporarily replaces the function during this block
    pass
```

---

### Best Practices

* Use only when you **control the execution context** (e.g., your own app or test environment).
* Always **document monkey patches** clearly.
* Consider using **mocks** or **custom subclasses** where possible.
* Limit monkey patching to **testing or emergencies**, not production logic.

---

### Drawbacks

* Hard to trace or debug.
* Not future-proof — changes in the library may break the patch.
* Can lead to side-effects if used improperly.

---

### Summary

| Feature      | Detail                                         |
| ------------ | ---------------------------------------------- |
| Purpose      | Modify/extend behavior at runtime              |
| Flexibility  | High, but risky                                |
| Common Use   | Testing, patching third-party code             |
| Safer Option | `unittest.mock.patch`, decorators, subclassing |
| Risk         | Hidden errors, maintenance overhead            |

---
