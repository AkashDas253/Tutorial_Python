## **Asyncio in Python**  

**`asyncio`** is a Python module for writing **asynchronous**, **non-blocking** code using `async` and `await`. It is useful for I/O-bound tasks like network requests, database queries, or file operations where waiting is required.

---

## **1. Basic Syntax of `asyncio`**  

```python
import asyncio

async def function_name():
    # Asynchronous task

asyncio.run(function_name())  # Runs the async function
```

### **Example**  
```python
import asyncio

async def say_hello():
    print("Hello,")
    await asyncio.sleep(1)  # Simulates a non-blocking delay
    print("World!")

asyncio.run(say_hello())
```
📌 **`await asyncio.sleep(1)` pauses execution without blocking other tasks.**

---

## **2. Running Multiple Async Tasks**  

```python
import asyncio

async def task(name, delay):
    await asyncio.sleep(delay)
    print(f"Task {name} completed")

async def main():
    await asyncio.gather(task("A", 2), task("B", 1))

asyncio.run(main())
```
📌 **`asyncio.gather()` runs multiple tasks concurrently.**

---

## **3. Using `asyncio.create_task()` for Background Tasks**  
```python
import asyncio

async def background_task():
    await asyncio.sleep(2)
    print("Background task done")

async def main():
    task = asyncio.create_task(background_task())  # Runs in background
    print("Main function continues")
    await asyncio.sleep(1)
    print("Main function done")
    await task  # Wait for the background task to finish

asyncio.run(main())
```

---

## **4. Using an Async Queue for Task Management**  
```python
import asyncio

async def worker(q):
    while not q.empty():
        item = await q.get()
        print(f"Processing {item}")
        await asyncio.sleep(1)
        q.task_done()

async def main():
    q = asyncio.Queue()
    for i in range(5):
        await q.put(i)

    workers = [asyncio.create_task(worker(q)) for _ in range(2)]
    await q.join()

asyncio.run(main())
```
📌 **`asyncio.Queue()` allows safe communication between async tasks.**

---

## **5. Async Context Managers**  
```python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("Releasing resource")

async def main():
    async with AsyncResource() as res:
        print("Using resource")

asyncio.run(main())
```
📌 **`async with` ensures proper resource management for async operations.**

---

## **6. Using `asyncio.Lock()` for Synchronization**  
```python
import asyncio

lock = asyncio.Lock()

async def task(n):
    async with lock:
        print(f"Task {n} executing")
        await asyncio.sleep(1)

async def main():
    await asyncio.gather(task(1), task(2), task(3))

asyncio.run(main())
```
📌 **`asyncio.Lock()` prevents race conditions in shared resources.**

---
