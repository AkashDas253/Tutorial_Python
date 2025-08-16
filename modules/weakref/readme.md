# `weakref` Module in Python

### Overview

* The **`weakref` module** provides tools for creating **weak references** to objects.
* A **weak reference** does **not increase the reference count** of the object.
* Useful when you want to reference objects **without preventing their garbage collection**.

---

### Key Features

* Allows **weak referencing** of objects that support it (not all types, e.g., `int` and `str` usually don’t).
* Provides **`weakref.ref`**, **`WeakKeyDictionary`**, **`WeakValueDictionary`**, and **`WeakSet`**.
* Commonly used in **caching, object graphs, and observer patterns** where you don’t want objects to stay alive unnecessarily.

---

### Syntax & Usage

#### 1. Creating a Weak Reference

```python
import weakref

class Data:
    pass

obj = Data()
weak = weakref.ref(obj)  # weak reference to obj

print(weak())   # Access object (if alive)

del obj
print(weak())   # None, since object was garbage collected
```

---

#### 2. `WeakValueDictionary`

Stores objects as values with weak references. If an object is collected, it disappears from the dictionary.

```python
import weakref

class Cache:
    pass

cache = weakref.WeakValueDictionary()
obj = Cache()

cache['item'] = obj
print(cache.get('item'))  # Cache object

del obj
print(cache.get('item'))  # None, since object was GC'd
```

---

#### 3. `WeakKeyDictionary`

Stores objects as keys with weak references. If the key object is collected, the entry is removed.

```python
import weakref

class User:
    pass

u = User()
data = weakref.WeakKeyDictionary()
data[u] = "active"

print(list(data.items()))  # [(<User object>, 'active')]

del u
print(list(data.items()))  # [] (entry removed automatically)
```

---

#### 4. `WeakSet`

A set that holds weak references. Objects disappear automatically when collected.

```python
import weakref

class Task:
    pass

t = Task()
tasks = weakref.WeakSet([t])

print(list(tasks))  # [<Task object>]

del t
print(list(tasks))  # [] (auto removed)
```

---

### Summary Table

| WeakRef Type          | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| `weakref.ref`         | Simple weak reference to an object                                  |
| `WeakValueDictionary` | Values are weak references; vanish when objects are GC’d            |
| `WeakKeyDictionary`   | Keys are weak references; entries auto-removed when keys are GC’d   |
| `WeakSet`             | Stores objects with weak references; objects auto-removed when GC’d |

---
