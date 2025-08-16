# Iterators in Python 

## Concept

* **Iterator** is an object in Python that represents a stream of data, producing items one at a time.
* Follows **Iterator Protocol** → requires two methods:

  * `__iter__()` → returns the iterator object itself.
  * `__next__()` → returns the next element; raises `StopIteration` when no items remain.
* Foundation for **for-loops**, **comprehensions**, and **generator constructs**.

---

## Key Characteristics

* Provides sequential access to elements.
* Consumed once → cannot be reset (unless recreated).
* Enables lazy evaluation, working well with large or infinite data.
* Forms the basis for **iterables** (objects that can return an iterator).

---

## Syntax

### Built-in Iterator Usage

```python
nums = [1, 2, 3]
it = iter(nums)        # Get iterator
print(next(it))        # 1
print(next(it))        # 2
print(next(it))        # 3
```

### Custom Iterator

```python
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            return self.current
        else:
            raise StopIteration

for num in Counter(3):
    print(num)   # 1, 2, 3
```

---

## Relation to Generators

* **Iterator**: Any object implementing `__iter__()` and `__next__()`.
* **Generator**: A simpler way to create iterators using `yield`.
* Every **generator is an iterator**, but not every iterator is a generator.

---

## Usage Scenarios

* Sequential data traversal without loading everything in memory.
* Custom data streams (infinite or dynamic).
* Basis for comprehension constructs (`list`, `set`, `dict`, generator expressions).
* Used heavily in libraries like `itertools` for functional-style programming.

---
