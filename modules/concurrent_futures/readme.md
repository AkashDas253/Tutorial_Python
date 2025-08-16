# `concurrent.futures` Module

The `concurrent.futures` module provides a **high-level interface for asynchronous execution** of tasks using threads or processes. It abstracts the lower-level thread and multiprocessing management into **executors** and **futures**.

---

## Core Components

### Executor Classes

* **ThreadPoolExecutor**

  * Uses threads from a pool to execute tasks.
  * Suitable for **I/O-bound** tasks.
* **ProcessPoolExecutor**

  * Uses processes from a pool to execute tasks.
  * Suitable for **CPU-bound** tasks.

### Future Object

Represents the result of an asynchronous computation.

* Methods:

  * `.result(timeout=None)` → returns result when available.
  * `.done()` → checks if task finished.
  * `.cancel()` → cancels task if not yet started.
  * `.running()` → checks if task is running.
  * `.exception(timeout=None)` → returns exception raised, if any.
  * `.add_done_callback(fn)` → attach a callback function when task completes.

---

## Key Functions

* \*\*submit(fn, \*args, **kwargs)**

  * Submits a callable to the executor for execution.
  * Returns a `Future` object.
* \**map(func, *iterables, timeout=None, chunksize=1)**

  * Like built-in `map()`, but runs in parallel.
  * Returns an iterator of results.
* **as\_completed(fs, timeout=None)**

  * Yields futures as they complete.
* **wait(fs, timeout=None, return\_when=ALL\_COMPLETED)**

  * Blocks until futures finish.
  * `return_when` options:

    * `FIRST_COMPLETED`
    * `FIRST_EXCEPTION`
    * `ALL_COMPLETED`

---

## Usage Scenarios

* **ThreadPoolExecutor**

  * Network requests
  * File I/O
  * Database queries
* **ProcessPoolExecutor**

  * Numerical computations
  * Image processing
  * Machine learning preprocessing

---

## Examples

### Example 1: Using `ThreadPoolExecutor`

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(task, i) for i in range(5)]
    for f in futures:
        print(f.result())
```

### Example 2: Using `map`

```python
from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

with ProcessPoolExecutor() as executor:
    results = executor.map(square, range(5))
    print(list(results))
```

### Example 3: Using `as_completed`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def delayed_square(x):
    time.sleep(x)
    return x * x

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(delayed_square, i) for i in range(5)]
    for f in as_completed(futures):
        print(f.result())
```

### Example 4: Handling Exceptions

```python
from concurrent.futures import ThreadPoolExecutor

def faulty_task(x):
    if x == 2:
        raise ValueError("Bad value")
    return x

with ThreadPoolExecutor() as executor:
    future = executor.submit(faulty_task, 2)
    try:
        print(future.result())
    except Exception as e:
        print("Caught:", e)
```

---
