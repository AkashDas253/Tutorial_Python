## Python Comprehensions 

### Overview

* **Definition**: Compact syntax for creating sequences (lists, sets, dictionaries, generators) from existing iterables.
* **Purpose**: Improves readability, conciseness, and functional-style programming.
* **Types**:

  * **List Comprehension** — produces a list.
  * **Set Comprehension** — produces a set (unique elements).
  * **Dictionary Comprehension** — produces a dictionary.
  * **Generator Expression** — produces a generator (lazy evaluation).

---

### General Syntax

#### List Comprehension

```python
[expression for item in iterable if condition]
```

* `expression` — value to include in the result.
* `item` — variable representing each element from `iterable`.
* `iterable` — any iterable object (list, tuple, range, etc.).
* `condition` (optional) — filter elements.

#### Set Comprehension

```python
{expression for item in iterable if condition}
```

#### Dictionary Comprehension

```python
{key_expr: value_expr for item in iterable if condition}
```

#### Generator Expression

```python
(expression for item in iterable if condition)
```

---

### Features & Usage Notes

* **Supports multiple `for` loops** for nested iterations.
* **Supports multiple `if` conditions**.
* **Generator expressions** save memory (evaluated lazily).
* **Comprehensions can replace `map()` and `filter()`** in many cases.
* **Avoid overly complex comprehensions** for readability.

---

### Examples

#### List Comprehension

```python
# Squares of even numbers
nums = [1, 2, 3, 4, 5, 6]
squares_even = [n**2 for n in nums if n % 2 == 0]
print(squares_even)  # [4, 16, 36]
```

#### Set Comprehension

```python
# Unique lengths of words
words = ["apple", "banana", "cherry", "apple"]
unique_lengths = {len(word) for word in words}
print(unique_lengths)  # {5, 6}
```

#### Dictionary Comprehension

```python
# Mapping numbers to their squares
nums = range(5)
squares_dict = {n: n**2 for n in nums}
print(squares_dict)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

#### Generator Expression

```python
# Sum of squares of even numbers
nums = [1, 2, 3, 4, 5, 6]
sum_squares_even = sum(n**2 for n in nums if n % 2 == 0)
print(sum_squares_even)  # 56
```

#### Nested Comprehension

```python
# Flatten a 2D list
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6]
```

---
