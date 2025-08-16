# Coroutines in Python

## Concept

* Coroutines are generalized functions that can pause and resume their execution at certain points.
* Unlike generators (which are data producers), coroutines are primarily **data consumers** but can also produce values.
* Built on Python's generator mechanism with `yield`, extended with `send()`, `throw()`, and `close()`.
* Often used in asynchronous programming, cooperative multitasking, event-driven systems.

---

## Key Characteristics

* **Defined with `def` and `yield`** (traditional) or **`async def`** (modern async/await style).
* Can **consume data** using `send()` instead of just yielding.
* Support **exception handling inside coroutines** with `throw()`.
* Can be **closed explicitly** with `close()`.
* Used in frameworks like `asyncio` for concurrency.

---

## Syntax

### Generator-based Coroutine

```python
def simple_coroutine():
    print("Coroutine started")
    x = yield  # Waits for a value
    print("Received:", x)

coro = simple_coroutine()
next(coro)          # Start coroutine (runs until first yield)
coro.send(10)       # Send value to coroutine
coro.close()        # Close coroutine
```

### Async/Await Coroutine (Modern Style)

```python
import asyncio

async def async_coro():
    print("Start async task")
    await asyncio.sleep(1)  # Non-blocking pause
    print("End async task")

asyncio.run(async_coro())
```

---

## Important Methods

* `next(coro)` → Starts or resumes coroutine until next `yield`.
* `coro.send(value)` → Sends data to coroutine at `yield`.
* `coro.throw(exc)` → Raises an exception inside coroutine.
* `coro.close()` → Closes coroutine gracefully.

---

## Use Cases

* Asynchronous I/O (networking, file handling).
* Event-driven applications.
* Pipelines for data processing.
* Cooperative multitasking.

---
