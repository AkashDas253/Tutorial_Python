# Strong References in Python

### Overview

* A **strong reference** is the default way Python variables refer to objects.
* When you assign an object to a variable, it creates a strong reference.
* **Strong references increase the reference count** of the object, keeping it alive in memory until all strong references are removed.

---

### Key Points

* As long as at least one strong reference exists, the object **cannot** be garbage-collected.
* When all strong references are deleted, the object’s reference count becomes zero, and Python’s garbage collector frees it.
* Most Python code works with strong references unless explicitly using **weak references**.

---

### Example: Strong Reference

```python
class Data:
    pass

obj = Data()     # strong reference
ref = obj        # another strong reference

print(ref is obj)  # True, both point to the same object

del obj
print(ref)        # Still valid, because `ref` is a strong reference
```

Here, `ref` keeps the object alive even after `obj` is deleted.

---

### Strong vs Weak References (Quick Table)

| Feature                | Strong Reference                | Weak Reference                           |
| ---------------------- | ------------------------------- | ---------------------------------------- |
| Reference count impact | Increases count                 | Does not increase count                  |
| Garbage collection     | Object alive until all are gone | Object can be collected anytime          |
| Default in Python      | Yes                             | No (requires `weakref` module)           |
| Usage                  | Normal variable assignments     | Caches, observers, memory-sensitive data |

---
