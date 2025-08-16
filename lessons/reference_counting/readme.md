# Reference Counting in Python

### Overview

* Reference counting is Python’s primary memory management technique.
* Each object keeps track of the number of references (pointers) that refer to it.
* When the reference count drops to **zero**, the object is immediately deallocated (freed from memory).
* Python also uses **garbage collection** (GC) to handle cycles that reference counting alone can’t manage.

---

### How It Works

* **Creation** → When an object is created, its reference count starts at `1`.
* **Assignment** → Assigning the object to another variable increases the count.
* **Function Calls** → Passing an object to a function increases the count.
* **Deletion** → Using `del` or removing references decreases the count.
* **Deallocation** → When the count hits `0`, memory is freed.

---

### Functions to Check Reference Count

* `sys.getrefcount(obj)` → Returns the current reference count of an object.

  * Note: It returns **+1** because `obj` is also passed as an argument.

---

### Example

```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # Reference count (usually 2: one for 'a' and one for argument)

b = a
print(sys.getrefcount(a))  # Increased (because 'b' references same object)

del b
print(sys.getrefcount(a))  # Decreased (reference from 'b' removed)

del a  # Now reference count = 0 → Object deallocated automatically
```

---

### Issues with Reference Counting

* **Circular References**: Objects referencing each other may never reach zero count.

  * Example: A linked list node pointing to itself.
  * Python’s **`gc` module** helps detect and clean such cycles.

---

### Usage Scenarios

* Understanding memory management in Python.
* Debugging memory leaks.
* Writing efficient, memory-safe applications.

---
