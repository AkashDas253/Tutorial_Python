# Generators in Python

## Concept

* **Generators** are a type of iterable in Python, like lists or tuples, but they generate values on the fly instead of storing them in memory.
* They are memory-efficient, suitable for working with large datasets or streams of data.
* Built using **generator functions** (with `yield`) or **generator expressions** (like list comprehensions but with parentheses).
* Execution is paused and resumed, maintaining state between yields.

---

## Key Characteristics

* **Lazy evaluation** → values are produced only when requested.
* **Stateful execution** → remembers where it left off.
* **Single-use** → can be iterated only once.
* **Efficiency** → avoids loading entire data into memory.

---

## Syntax

### Generator Function

```python
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
```

### Generator Expression

```python
gen_exp = (x**2 for x in range(5))
for val in gen_exp:
    print(val)   # 0, 1, 4, 9, 16
```

---

## Built-in Functions with Generators

* `next(generator)` → fetch next item.
* `send(value)` → send a value into generator.
* `throw(exception)` → raise exception inside generator.
* `close()` → stop generator.

---

## Usage Scenarios

* **Processing large files/streams** (e.g., reading line by line).
* **Pipelines** (chaining multiple data transformations).
* **Infinite sequences** (e.g., Fibonacci, random numbers).
* **Asynchronous programming** (with `async def` and `await`).

---
