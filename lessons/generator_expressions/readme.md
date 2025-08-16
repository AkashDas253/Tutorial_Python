# Generator Expressions in Python

### Concept

* A **generator expression** is a concise way to create generators using a syntax similar to list comprehensions but with **parentheses** instead of square brackets.
* They **produce values lazily**, generating items one at a time instead of creating an entire collection in memory.
* Useful for handling large datasets, streaming data, or infinite sequences.

---

### Characteristics

* **Lazy Evaluation**: Values computed only when requested.
* **Memory Efficient**: Does not store the whole sequence in memory.
* **Single Use**: Once exhausted, cannot be reused.
* **Syntactic Sugar**: Shorter form of writing generator functions.
* **Supports Iteration**: Can be iterated with `for` loops, `next()`, or functions like `sum()`, `any()`, `all()`.

---

### Syntax

```python
# Generator expression syntax
(expression for item in iterable if condition)
```

* `expression` → Value to yield each time.
* `item` → Variable representing each element.
* `iterable` → Source of data.
* `condition` (optional) → Filters elements.

---

### Examples

```python
# Basic generator expression
gen = (x * 2 for x in range(5))
print(next(gen))   # 0
print(next(gen))   # 2
print(list(gen))   # [4, 6, 8]

# Using with sum
total = sum(x*x for x in range(5))  # 0²+1²+2²+3²+4² = 30
print(total)

# With condition
even_squares = (x*x for x in range(10) if x % 2 == 0)
print(list(even_squares))  # [0, 4, 16, 36, 64]

# Nested generator expression
pairs = ((x, y) for x in range(3) for y in range(2))
print(list(pairs))  # [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
```

---

### Usage Scenarios

* When working with **large datasets** where list comprehensions would be memory-heavy.
* For **stream processing** where data is consumed as it arrives.
* As **arguments to functions** that consume iterables (`sum()`, `min()`, `max()`, `any()`, `all()`).
* Efficient alternative to list comprehensions when the **final list is not needed**.

---
