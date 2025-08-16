# File Handling – `pathlib`

### Overview

* `pathlib` provides an **object-oriented interface** for file system paths.
* Introduced in Python 3.4, it serves as a modern alternative to `os` and `os.path` for file and directory handling.
* Uses classes like `Path` to represent filesystem paths and perform file operations.
* Cross-platform support (handles `WindowsPath` and `PosixPath` automatically).
* Encourages **cleaner, chainable, and readable code** compared to procedural `os` functions.

---

### Key Features

* Unified interface for **paths**, whether files or directories.
* Methods for **path manipulation** (joining, splitting, resolving).
* File operations: read, write, append, touch, unlink.
* Directory operations: create, remove, iterate, glob patterns.
* Metadata access (size, permissions, timestamps).
* Integration with context managers for safe resource handling.

---

### Syntax and Usage

```python
from pathlib import Path

# Define a path
p = Path("example.txt")

# Reading
content = p.read_text(encoding="utf-8")  

# Writing (overwrites if exists)
p.write_text("Hello, pathlib!", encoding="utf-8")  

# Appending
with p.open(mode="a", encoding="utf-8") as f:
    f.write("\nAppending new line")

# File iteration
for line in p.open(mode="r", encoding="utf-8"):
    print(line.strip())

# Path operations
print(p.name)        # 'example.txt'
print(p.suffix)      # '.txt'
print(p.stem)        # 'example'
print(p.parent)      # parent directory
print(p.exists())    # True if file exists

# Directory operations
d = Path("my_dir")
d.mkdir(exist_ok=True)  
for file in d.iterdir():
    print(file)

# Glob patterns
for pyfile in Path(".").rglob("*.py"):
    print(pyfile)
```

---

### File Handling with Context Manager (`pathlib`)

* Like `os`, `pathlib.Path.open()` integrates with `with` for safer handling:

```python
with Path("data.txt").open(mode="w", encoding="utf-8") as f:
    f.write("Safe writing with context manager")
```

---

### Advantages over `os`

* Object-oriented, less error-prone.
* Cleaner chaining: `Path("dir") / "file.txt"` instead of `os.path.join("dir", "file.txt")`.
* Built-in read/write functions (no need for explicit `open` unless special modes required).
* Automatically adjusts to OS-specific path styles.

---
