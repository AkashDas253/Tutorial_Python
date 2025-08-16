# Weak References in Python

### Overview

* A **weak reference** allows one object to reference another **without increasing its reference count**.
* Useful for caching, tracking objects, or preventing memory leaks caused by circular references.
* Implemented in Python via the **`weakref` module**.

---

### Key Points

* **Normal reference** → Increases reference count, object won’t be freed until all references are gone.
* **Weak reference** → Does not increase reference count, so the object can still be garbage-collected.
* If the object is collected, the weak reference automatically becomes **dead** (returns `None`).

---

### `weakref` Module Features

* **`weakref.ref(obj)`** → Creates a weak reference to an object.
* **`weakref.proxy(obj)`** → Similar to `ref`, but returns a proxy object that behaves like the original.
* **`weakref.WeakKeyDictionary`** → Dictionary with weakly-referenced keys.
* **`weakref.WeakValueDictionary`** → Dictionary with weakly-referenced values.
* **`weakref.WeakSet`** → Set storing weak references.
* **Callbacks** → A function can be registered to execute when the referenced object is finalized.

---

### Example: Basic Weak Reference

```python
import weakref

class Data:
    pass

obj = Data()
weak = weakref.ref(obj)

print(weak())   # Access the object
del obj         # Delete strong reference
print(weak())   # Now None (object collected)
```

---

### Example: WeakValueDictionary

```python
import weakref

class User:
    def __init__(self, name):
        self.name = name

users = weakref.WeakValueDictionary()

u = User("Alice")
users["user1"] = u

print(users["user1"].name)  # Alice

del u  # Strong ref removed
print(users.get("user1"))   # None (object collected)
```

---

### Benefits

* Prevents memory leaks in caches, mappings, or observer patterns.
* Helps manage objects that should disappear when no strong references exist.

---

### Limitations

* Not all objects can be weakly referenced (e.g., `int`, `str`, `tuple`—immutable builtins).
* Weak references add slight overhead.

---
