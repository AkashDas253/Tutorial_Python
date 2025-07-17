## Low-Level Standard Input/Output in Python

### What is Low-Level I/O?

Low-level I/O in Python gives **finer control over input and output operations** by directly interacting with the **standard streams**:

* `sys.stdin`: Standard input stream (keyboard or piped)
* `sys.stdout`: Standard output stream (screen or redirected)
* `sys.stderr`: Standard error stream (used for logging errors)

These are file-like objects exposed by the `sys` module.

---

## Modules Used

```python
import sys
```

---

## `sys.stdin` – Low-Level Input

### Description:

Reads input directly from the **standard input stream**.

### Methods:

| Method                  | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `sys.stdin.read([n])`   | Reads **entire input** or up to `n` characters |
| `sys.stdin.readline()`  | Reads **one line** from input (includes `\n`)  |
| `sys.stdin.readlines()` | Reads all lines as a **list of strings**       |

### Examples:

```python
import sys

# Read entire input
data = sys.stdin.read()
print("Read data:", data)

# Read one line
line = sys.stdin.readline()
print("Line:", line)

# Read all lines
lines = sys.stdin.readlines()
print("Lines:", lines)
```

---

## `sys.stdout` – Low-Level Output

### Description:

Writes output to the **standard output stream** directly.

### Methods:

| Method                | Description                     |
| --------------------- | ------------------------------- |
| `sys.stdout.write(s)` | Writes string `s` to screen     |
| `sys.stdout.flush()`  | Forces flush of internal buffer |

### Examples:

```python
import sys

sys.stdout.write("This is low-level stdout\n")
sys.stdout.flush()  # Optional, forces immediate write
```

---

## `sys.stderr` – Standard Error Output

### Description:

Used to output **error messages** or **logs**, separate from normal output.

### Methods:

Same as `sys.stdout`.

### Examples:

```python
import sys

sys.stderr.write("This is an error message\n")
```

---

## Differences Between `print()` and Low-Level I/O

| Feature                    | `print()`                        | `sys.stdout.write()`       |
| -------------------------- | -------------------------------- | -------------------------- |
| Automatically adds newline | Yes (`\n`)                       | No                         |
| Type conversion            | Converts all types to string     | Requires manual conversion |
| Output buffering           | Buffered                         | Buffered                   |
| Stream control             | Limited                          | Full control               |
| Formatting                 | Built-in with `sep`, `end`, etc. | Must be manual             |

---

## Use Cases for Low-Level I/O

* Reading input from **files or pipes** in Unix-style scripting
* Writing output **without newlines** or formatting
* Outputting **error logs** separately using `sys.stderr`
* **Performance tuning** where explicit flushing is needed
* **Redirecting input/output** in command-line tools

---

## Redirection Example (in Shell)

You can redirect input/output at the OS level when using low-level I/O:

```bash
python myscript.py < input.txt > output.txt 2> error.log
```

| Symbol | Meaning                  |
| ------ | ------------------------ |
| `<`    | Redirect standard input  |
| `>`    | Redirect standard output |
| `2>`   | Redirect standard error  |

---

## Best Practices

* Use `sys.stdin` and `sys.stdout` for **bulk** or **streamed** input/output.
* Use `sys.stderr` for **diagnostic** or **error reporting**.
* Always call `.flush()` if you want **immediate visibility** of output.
* Avoid mixing `print()` and `sys.stdout.write()` unnecessarily to maintain consistency.

---
