## Python Pattern Matching 

### Overview

Pattern matching in Python is introduced with **`match`** and **`case`** statements (Python 3.10+), allowing structural pattern matching similar to `switch` statements in other languages but with more powerful data destructuring capabilities.

---

### Syntax

```python
match subject:
    case pattern_1:
        # Code block if pattern_1 matches
    case pattern_2:
        # Code block if pattern_2 matches
    case _:
        # Default case (matches anything)
```

---

### Key Concepts

* **Subject**
  The value being matched against patterns.

* **Patterns**
  Structures that the subject is tested against.

* **Wildcard Pattern `_`**
  Matches any value (similar to `default` case).

* **Literal Patterns**
  Match exact values.

* **Capture Patterns**
  Assign matched value to a variable.

* **Sequence Patterns**
  Match sequences like lists or tuples.

* **Mapping Patterns**
  Match dictionary-like structures.

* **Class Patterns**
  Match objects and extract attributes.

* **OR Patterns (`|`)**
  Match if any subpattern matches.

* **Guards (`if` clauses)**
  Add extra conditions for a pattern to match.

---

### Types of Patterns & Examples

#### Literal Pattern

```python
value = 42
match value:
    case 0:
        print("Zero")
    case 42:
        print("The Answer")
```

#### Capture Pattern

```python
match value:
    case x:
        print(f"Captured: {x}")
```

#### Sequence Pattern

```python
data = [1, 2, 3]
match data:
    case [1, 2, 3]:
        print("Exact match")
    case [1, *rest]:
        print(f"Starts with 1, rest: {rest}")
```

#### Mapping Pattern

```python
person = {"name": "Alice", "age": 30}
match person:
    case {"name": name, "age": age}:
        print(f"{name} is {age} years old")
```

#### Class Pattern

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(10, 20)

match p:
    case Point(x, y):
        print(f"Point at ({x}, {y})")
```

#### OR Pattern

```python
value = "yes"
match value:
    case "yes" | "y" | "ok":
        print("Confirmed")
```

#### Guarded Pattern

```python
value = 10
match value:
    case x if x > 5:
        print("Greater than 5")
```

---

### Usage Scenarios

* Simplifying complex `if-elif` chains
* Parsing structured data (like JSON)
* Handling multiple types or formats in a clean way
* Extracting and processing parts of data structures

---
