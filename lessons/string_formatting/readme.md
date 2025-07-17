## String Formatting in Python

Python provides **multiple string formatting mechanisms**, each suited for different use cases. String formatting allows for **inserting variables**, **aligning**, **padding**, **controlling precision**, and more.

---

## Categories of String Formatting

* **Old-style formatting** (`%` operator)
* **`str.format()` method**
* **Formatted string literals (f-strings)** – introduced in Python 3.6
* **Template strings** – from the `string` module (safe, for user input scenarios)

---

## 1. Old-Style Formatting (`%` operator)

### Format Specifiers

| Symbol | Meaning           |
| ------ | ----------------- |
| `%s`   | String            |
| `%d`   | Integer (decimal) |
| `%f`   | Floating-point    |
| `%x`   | Hexadecimal       |
| `%%`   | Literal `%`       |

### Syntax

```python
"Hello, %s" % "Alice"
"Value: %d" % 42
"Float: %.2f" % 3.14159
```

---

## 2. `str.format()` Method

More powerful and flexible than `%` formatting.

### Placeholders:

```python
"Hello, {}".format("Alice")
"{0} scored {1}".format("Alice", 90)
"{name} got {score}".format(name="Bob", score=95)
```

### Format Spec Mini-Language:

```python
"{:<10}".format("left")    # Left align
"{:>10}".format("right")   # Right align
"{:^10}".format("center")  # Center align
"{:.2f}".format(3.14159)   # Float with 2 decimal places
"{:0>5}".format(42)        # Pad with zeros
```

---

## 3. f-Strings (Formatted String Literals)

Introduced in Python 3.6, it's the most **concise** and **readable** way.

### Basic Usage:

```python
name = "Alice"
score = 95
f"{name} scored {score}"
```

### Advanced:

```python
f"{3.14159:.2f}"            # Precision formatting
f"{42:0>5}"                 # Zero-padding
f"{'hi':^10}"               # Center aligned
f"{name.upper()} got {score}"
```

---

## 4. `string.Template` (from `string` module)

Useful when dealing with **external input** (e.g., from users).

### Syntax:

```python
from string import Template
t = Template("Hello, $name! Your score is $score.")
t.substitute(name="Alice", score=90)
```

Safer: Raises errors if keys are missing, unlike f-strings.

---

## Comparison Summary

| Feature        | `%` Formatting | `str.format()`     | f-Strings | `Template`        |
| -------------- | -------------- | ------------------ | --------- | ----------------- |
| Style          | C-style        | Verbose            | Concise   | Simple, safe      |
| Python version | All            | 2.7+               | 3.6+      | All               |
| Readability    | Moderate       | Low                | High      | Moderate          |
| Best use case  | Legacy code    | Complex formatting | General   | Safe substitution |

---

## Advanced Options

### Numeric Formatting

```python
f"{1234.56789:,.2f}"      # '1,234.57'
"{:e}".format(12345678)   # Exponential
```

### Binary, Hex, Octal

```python
f"{42:b}"    # binary
f"{42:x}"    # hex
f"{42:o}"    # octal
```

---

## Common Use Cases

* Console output formatting
* Dynamic message construction
* Report generation
* Log formatting
* Template-based input filling (e.g., emails, HTML)

---
