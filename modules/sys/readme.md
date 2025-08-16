# `sys` Module in Python

The **`sys` module** provides access to variables and functions that interact closely with the Python interpreter. It is essential for controlling runtime environment, accessing command-line arguments, manipulating paths, and interpreter state.

---

## Key Features

* Interpreter information
* Command-line arguments
* System paths & module management
* Standard streams (stdin, stdout, stderr)
* Exception handling
* Recursion and execution limits
* Exit handling
* Dynamic import and runtime control

---

## Components

### System Information

* `sys.version` → Python version string
* `sys.version_info` → Version details tuple `(major, minor, micro, releaselevel, serial)`
* `sys.api_version` → C API version
* `sys.implementation` → Implementation details (e.g., CPython, PyPy)
* `sys.platform` → Platform identifier
* `sys.executable` → Path to Python interpreter

---

### Command-line Arguments

* `sys.argv` → List of arguments passed to the script
* `sys.flags` → Command-line flags (like `-O`, `-B`)

---

### Path and Import Control

* `sys.path` → List of directories Python searches for modules
* `sys.path_hooks` → Functions used for import path handling
* `sys.path_importer_cache` → Cache of path finders
* `sys.meta_path` → Custom import hook handlers
* `sys.modules` → Dictionary of loaded modules

---

### Standard Streams

* `sys.stdin` → Standard input
* `sys.stdout` → Standard output
* `sys.stderr` → Standard error
* `sys.__stdin__`, `sys.__stdout__`, `sys.__stderr__` → Original streams before redirection

---

### Exit & Exception Handling

* `sys.exit([code])` → Exit program
* `sys.exc_info()` → Info about the current exception `(type, value, traceback)`
* `sys.last_type`, `sys.last_value`, `sys.last_traceback` → Info on last unhandled exception
* `sys.excepthook` → Custom exception handler
* `sys.unraisablehook` → Handles unraisable exceptions

---

### Limits & Recursion

* `sys.getrecursionlimit()` / `sys.setrecursionlimit(n)` → Manage recursion depth
* `sys.getsizeof(object, default)` → Size of object in bytes
* `sys.getswitchinterval()` / `sys.setswitchinterval(interval)` → Thread switching interval

---

### Memory & Internals

* `sys.getrefcount(object)` → Reference count of an object
* `sys.intern(string)` → Interns a string for faster comparison
* `sys.getallocatedblocks()` → Number of memory blocks allocated
* `sys.gettotalrefcount` (debug builds)

---

### Interpreter State & Dynamic Control

* `sys.getdefaultencoding()` → Default encoding (`utf-8`)
* `sys.getfilesystemencoding()` → Encoding for file system paths
* `sys.dont_write_bytecode` → Prevents writing `.pyc` files
* `sys.getprofile()` / `sys.setprofile(func)` → Profiling hooks
* `sys.gettrace()` / `sys.settrace(func)` → Debugging trace hooks

---

### Miscellaneous

* `sys.byteorder` → Endianness (`little` or `big`)
* `sys.maxsize` → Maximum size lists/strings can take (platform dependent)
* `sys.maxunicode` → Maximum Unicode code point
* `sys.copyright` → Copyright string
* `sys.builtin_module_names` → Tuple of built-in module names

---

## Examples

```python
import sys

# Command-line arguments
print("Arguments:", sys.argv)

# System information
print("Python version:", sys.version)
print("Platform:", sys.platform)
print("Interpreter:", sys.executable)

# Recursion limit
print("Recursion limit:", sys.getrecursionlimit())

# Memory & references
x = []
print("Ref count of x:", sys.getrefcount(x))
print("Size of list x:", sys.getsizeof(x), "bytes")

# Exit example
try:
    sys.exit(1)
except SystemExit as e:
    print("Exited with code:", e.code)

# Redirect stdout
sys.stdout.write("Hello without print()\n")
```

---
