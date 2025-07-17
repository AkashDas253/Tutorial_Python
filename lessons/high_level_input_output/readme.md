## Standard High-Level Input and Output in Python

### What is Standard Input/Output?

Standard Input/Output (I/O) refers to interaction between the **user and the program via the console (terminal)**:

* **Standard Input (`stdin`)**: Keyboard input
* **Standard Output (`stdout`)**: Printed text on the screen
* **Standard Error (`stderr`)**: Error messages printed on screen

These are handled in Python using high-level functions like `input()` and `print()`.

---

## Standard Input (`input()`)

### Description:

Reads a line of text from **standard input (keyboard)** and returns it as a **string**.

### Syntax:

```python
input(prompt=None)
```

| Parameter | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| `prompt`  | *(Optional)* A string message displayed to the user before input. |

### Returns:

* Always returns a string.
* Must be explicitly converted to `int`, `float`, etc., if needed.

### Examples:

```python
name = input("Enter your name: ")
age = int(input("Enter your age: "))
```

---

## Standard Output (`print()`)

### Description:

Prints the given objects to the **standard output (screen)**.

### Syntax:

```python
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
```

| Parameter  | Description                                                   |
| ---------- | ------------------------------------------------------------- |
| `*objects` | One or more values to be printed.                             |
| `sep`      | *(Default: `' '`)* Separator between objects.                 |
| `end`      | *(Default: `'\n'`)* Appended after last object.               |
| `file`     | *(Default: `sys.stdout`)* Output destination.                 |
| `flush`    | *(Default: `False`)* If `True`, forcibly flush output buffer. |

### Examples:

```python
print("Hello", "World")                     # Hello World
print("A", "B", "C", sep="-")               # A-B-C
print("Loading...", end="")                 # no newline
print("Done!")
```

---

## Formatted Output

Python supports multiple ways to format output:

### f-Strings (Python 3.6+)

```python
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")
```

### `str.format()` Method

```python
print("Name: {}, Age: {}".format("Bob", 25))
print("Name: {n}, Age: {a}".format(n="Carol", a=28))
```

### `%` Operator (Old Style)

```python
print("Score: %.2f" % 95.678)        # Score: 95.68
print("Name: %s, Age: %d" % ("Eve", 32))
```

---

## Low-Level Redirection (still part of standard I/O)

Python uses `sys` module for lower-level I/O control.

### Syntax:

```python
import sys

sys.stdout.write("Output to stdout\n")
sys.stderr.write("Output to stderr\n")
data = sys.stdin.read()  # reads until EOF
```

---

## Best Practices

* Use `input()` with clear prompts.
* Use `print()` with `sep`, `end` and formatting when needed.
* Prefer `f-strings` for readability and performance.
* Use `sys.stdin` and `sys.stdout` only when needed for finer control or performance.

---

## Use Cases

| Task                 | Function           | Note                         |
| -------------------- | ------------------ | ---------------------------- |
| Read string input    | `input()`          | Always returns string        |
| Read multiple values | `input().split()`  | Needs parsing                |
| Print data           | `print()`          | Simple, high-level output    |
| Redirect output      | `file=sys.stderr`  | For errors or logs           |
| Continuous input     | `sys.stdin.read()` | For bulk or file piped input |

---
