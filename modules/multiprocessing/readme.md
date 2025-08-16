# Multiprocessing Module

The `multiprocessing` module provides **process-based parallelism** in Python, allowing you to bypass the Global Interpreter Lock (GIL) and fully utilize multiple CPU cores.

---

## Core Concepts

* **Process-based parallelism**: Each task runs in a separate Python process with its own memory space.
* **Communication**: Processes can communicate via `Queue`, `Pipe`, or shared memory.
* **Synchronization**: Locks, Semaphores, Events, Conditions are used to coordinate processes.
* **Process Pools**: Pool of worker processes for parallel execution.

---

## Key Classes

* **`multiprocessing.Process`**

  * Represents an independent process.
* **`multiprocessing.Pool`**

  * Manages multiple worker processes for parallel execution.
* **`multiprocessing.Queue`**

  * Thread/process-safe FIFO queue for communication.
* **`multiprocessing.Pipe`**

  * Two-way communication channel between two processes.
* **`multiprocessing.Manager`**

  * Provides shared objects like `list`, `dict`, `Namespace`.
* **`multiprocessing.Value` / `multiprocessing.Array`**

  * Shared memory variables for primitive types and arrays.

---

## Synchronization Primitives

* **Lock**: Ensures only one process accesses a resource at a time.
* **RLock**: Reentrant lock.
* **Semaphore**: Limits access to a resource pool.
* **BoundedSemaphore**: Semaphore with a maximum limit.
* **Event**: Signals processes to start/stop operations.
* **Condition**: Process waits until notified.

---

## Process Creation

```python
from multiprocessing import Process

def worker(name):
    print(f"Hello from {name}")

if __name__ == "__main__":
    p = Process(target=worker, args=("Process-1",))
    p.start()
    p.join()
```

---

## Process Pools

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as pool:   # 4 worker processes
        results = pool.map(square, [1, 2, 3, 4, 5])
        print(results)
```

---

## Communication with Queue

```python
from multiprocessing import Process, Queue

def producer(q):
    q.put("Hello from producer")

def consumer(q):
    msg = q.get()
    print("Consumer received:", msg)

if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))
    p1.start(); p2.start()
    p1.join(); p2.join()
```

---

## Using Manager for Shared Objects

```python
from multiprocessing import Manager, Process

def worker(shared_list):
    shared_list.append("data")

if __name__ == "__main__":
    with Manager() as manager:
        shared = manager.list()
        processes = [Process(target=worker, args=(shared,)) for _ in range(3)]
        for p in processes: p.start()
        for p in processes: p.join()
        print(shared)
```

---

## Shared Memory (Value & Array)

```python
from multiprocessing import Value, Array, Process

def worker(val, arr):
    val.value += 1
    arr[0] = arr[0] * 2

if __name__ == "__main__":
    v = Value('i', 0)      # shared integer
    a = Array('i', [1, 2, 3])
    p = Process(target=worker, args=(v, a))
    p.start(); p.join()
    print(v.value, list(a))
```

---

## Synchronization Example

```python
from multiprocessing import Process, Lock

def worker(lock, n):
    with lock:
        print(f"Process {n} acquired lock")

if __name__ == "__main__":
    lock = Lock()
    processes = [Process(target=worker, args=(lock, i)) for i in range(3)]
    for p in processes: p.start()
    for p in processes: p.join()
```

---

## When to Use

* CPU-bound tasks (e.g., data processing, simulations).
* Tasks that benefit from true parallel execution.
* When GIL-free execution is required.

---
