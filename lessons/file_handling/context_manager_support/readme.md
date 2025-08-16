## File Handling – Context Manager

A **context manager** in file handling provides a safe and convenient way to manage resources (like file objects) by automatically handling **setup and cleanup** tasks. In Python, the `with` statement is used with context managers.

---

### Key Points

* Ensures the file is **automatically closed** after operations, even if an exception occurs.
* Reduces boilerplate code compared to manual `open()` and `close()`.
* Improves **readability** and **safety** in file handling.
* File objects implement the context manager protocol (`__enter__` and `__exit__`).

---

### Syntax

```python
# Basic usage
with open("example.txt", "r") as file:
    data = file.read()

# Multiple context managers in one statement
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    for line in infile:
        outfile.write(line.upper())
```

---

### How It Works Internally

* `__enter__()` → Called when entering the context (returns file object).
* `__exit__()` → Called when exiting the context (closes the file, handles exceptions).

---

### Advantages

* No need to explicitly call `file.close()`.
* Prevents resource leaks.
* Cleaner, more Pythonic code.
* Exception safety: file is closed even if an error occurs.

---

### Usage Scenarios

* Reading or writing files where reliability is critical.
* Working with multiple files simultaneously.
* Handling files in programs that may raise runtime errors.

---
