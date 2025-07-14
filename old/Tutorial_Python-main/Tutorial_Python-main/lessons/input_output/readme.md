# Input and Output in Python

---

### **Overview**

Python provides built-in functions for basic **Input** and **Output** operations. Input is commonly performed using `input()` and output using `print()`. For more advanced needs, Python offers formatted printing, file-based I/O, and the `sys` module for lower-level operations.

---

## **Input**

---

### `input(prompt=None)`

Reads a line from standard input and returns it as a string.

- `prompt`: *(Optional)* A string displayed before user input. If omitted, no prompt is shown.

**Returns**: Always returns a string (use `int()`, `float()`, etc. to convert as needed).

```py
name = input("Enter your name: ")
print("Hello,", name)
```

---

## **Output**

---

### `print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)`

Used to print output to standard output or other writable streams.

- `*objects`: One or more objects to print, separated by `sep`.
- `sep`: *(Default: `' '`)* String inserted between objects.
- `end`: *(Default: `'\n'`)* String added after the final value.
- `file`: *(Default: `sys.stdout`)* A file-like object (stream) where output is sent.
- `flush`: *(Default: `False`)* If `True`, flushes the output buffer immediately.

```py
x = 10
y = 20
print("The sum is", x + y)  # default sep=' ', end='\n'
print(x, y, sep=' -> ', end=' !!!\n')
```

---

## **Formatted Output**

Python offers **formatted string literals** (f-strings), **`str.format()`**, and the **`%` operator** for formatted output.

---

### `str.format(*args, **kwargs)`

Used to create formatted strings.

- `*args`: Positional arguments for format placeholders.
- `**kwargs`: Keyword arguments for named placeholders.

```py
name = "Alice"
score = 95
print("Name: {}, Score: {}".format(name, score))
print("Name: {n}, Score: {s}".format(n=name, s=score))
```

---

### `format(value, format_spec='')`

Applies formatting to a single value using a format specification.

- `value`: The value to format.
- `format_spec`: The format string defining presentation (e.g., width, precision).

```py
pi = 3.14159
print(format(pi, ".2f"))       # 2 decimal places
print(format(42, "04"))        # padded with zeros: 0042
```

---

## **String Formatting Methods**

---

### **f-Strings (Python 3.6+)**

Prefixed with `f`, allows embedding expressions inside string literals using `{}`.

```py
name = "Bob"
age = 25
print(f"My name is {name} and I am {age} years old.")
```

---

### `%` Formatting

Uses C-style formatting operators, e.g., `%s`, `%d`, `%f`.

```py
score = 90.5
print("Score: %.2f" % score)
print("Name: %s, Age: %d" % ("Charlie", 30))
```

---

## **Low-level I/O**

---

### `sys.stdin.read([n])`, `sys.stdout.write(s)`, `sys.stderr.write(s)`

From the `sys` module (must be imported):

- `sys.stdin.read([n])`: Reads `n` characters or until EOF.
- `sys.stdout.write(s)`: Writes the string `s` to standard output (no newline).
- `sys.stderr.write(s)`: Writes string to the standard error stream.

```py
import sys

# Writing directly
sys.stdout.write("This is stdout\n")
sys.stderr.write("This is stderr\n")

# Reading directly
# Uncomment if using real input
# data = sys.stdin.read()
# print("You typed:", data)
```

---

## **File I/O Functions** (Quick Reference)

Use the `open()` function for file operations, covered in detail under File Handling.

---

### `open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)`

- `file`: Path or file name to open.
- `mode`: File mode (e.g., `'r'`, `'w'`, `'a'`, `'rb'`, `'wb'`).
- `buffering`: Buffer policy (`0`, `1`, or larger integer).
- `encoding`: Text encoding (e.g., `'utf-8'`).
- `errors`: Error handling (`'strict'`, `'ignore'`, etc.).
- `newline`: Controls newline translation.
- `closefd`: If `False`, keeps file descriptor open when closing file.
- `opener`: Custom opener for opening the file.

```py
# Write to a file
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("Hello File\n")

# Read from a file
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)
```

---
