# Context Managers in Python 

A **context manager** in Python is used to **manage resources** (like files, network connections, locks) **safely**.
It **automatically handles setup and cleanup**, ensuring resources are released even if exceptions occur.

---

## Key Features

* Ensures **deterministic resource management**.
* Commonly used with **`with` statements**.
* Provides **`__enter__()`** and **`__exit__()`** methods for setup and teardown.
* Prevents **resource leaks**.

---

## Syntax – Using `with` Statement

```python
with <context_manager> as <variable>:
    # block of code using the resource
```

Example with a file:

```python
with open("file.txt", "w") as f:
    f.write("Hello, World!")
# File is automatically closed after the block
```

---

## How Context Managers Work

* **`__enter__(self)`** → Called at the start of the `with` block. Returns a value (optional).
* **`__exit__(self, exc_type, exc_value, traceback)`** → Called at the end of the block. Handles cleanup.

  * Parameters indicate if an exception occurred:

    * `exc_type`, `exc_value`, `traceback`
  * Can suppress exceptions if needed by returning `True`.

---

## Creating a Custom Context Manager

### Using a Class

```python
class MyContext:
    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context")
        if exc_type:
            print(f"Exception: {exc_value}")
        return False  # Do not suppress exceptions

with MyContext() as ctx:
    print("Inside context")
```

Output:

```
Entering context
Inside context
Exiting context
```

---

### Using `contextlib` – Decorator Approach

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("Entering")
    yield "Resource"  # value returned by 'as' variable
    print("Exiting")

with my_context() as res:
    print("Using", res)
```

Output:

```
Entering
Using Resource
Exiting
```

---

## Handling Exceptions

```python
class SafeDivide:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Handled exception: {exc_value}")
            return True  # Suppress exception

with SafeDivide() as sd:
    result = 10 / 0
print("Program continues")  # Execution continues because exception was suppressed
```

---

## Use Cases

* File handling (`open`)
* Thread or process locks (`threading.Lock`)
* Database connections
* Network connections
* Temporary resource allocation

---

## Quick Syntax Summary

| Approach        | Methods                     | Example                  |
| --------------- | --------------------------- | ------------------------ |
| Class-based     | `__enter__()`, `__exit__()` | `with MyContext(): ...`  |
| Decorator-based | `@contextmanager`, `yield`  | `with my_context(): ...` |

---
