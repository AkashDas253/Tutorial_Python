# Iterables in Python 

## Concept

* **Iterable**: Any Python object capable of returning its elements one at a time.
* It must implement either:

  * `__iter__()` → returns an **iterator** object, or
  * `__getitem__()` with integer indexing (legacy style).
* **for-loops**, **comprehensions**, and many built-in functions work on iterables.

---

## Key Characteristics

* Iterable ≠ Iterator (an **iterable produces iterators**).
* Can be traversed multiple times (each call to `iter()` creates a new iterator).
* Provides a uniform interface for collections and sequences.
* Supports iteration without exposing internal structure.

---

## Examples of Iterables

* **Built-in collections**: `list`, `tuple`, `str`, `set`, `dict`, `range`.
* **File objects** (line by line).
* **Custom classes** implementing `__iter__()`.

---

## Syntax

### Basic Iterable

```python
nums = [10, 20, 30]   # list is iterable
for n in nums:
    print(n)          # 10, 20, 30
```

### Converting Iterable to Iterator

```python
nums = [1, 2, 3]
it = iter(nums)       # Get an iterator
print(next(it))       # 1
print(next(it))       # 2
print(next(it))       # 3
```

### Custom Iterable

```python
class Letters:
    def __init__(self, word):
        self.word = word
    
    def __iter__(self):
        return iter(self.word)   # delegate to string iterator

for ch in Letters("Hi"):
    print(ch)   # H, i
```

---

## Iterables vs Iterators

| Feature            | Iterable                                | Iterator                             |
| ------------------ | --------------------------------------- | ------------------------------------ |
| Definition         | Object capable of returning an iterator | Object that actually produces values |
| Method Requirement | `__iter__()` (or `__getitem__()`)       | `__iter__()` and `__next__()`        |
| Reset              | Can be restarted with new iterator      | Consumed once, cannot be restarted   |
| Examples           | list, tuple, str, set, dict, range      | result of `iter(list)`, generators   |

---

## Usage Scenarios

* General looping in Python (`for`, comprehensions).
* Input sources (files, streams, APIs).
* Foundation for advanced iteration (generators, coroutines).
* Enables lazy pipelines (`map`, `filter`, `zip`, `enumerate`).

---
