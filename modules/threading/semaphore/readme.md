# Thread Synchronization: Semaphore

A `Semaphore` is a synchronization primitive that controls access to a resource with a fixed number of slots.

## Syntax
```python
import threading
semaphore = threading.Semaphore(value)
```

## Usage
```python
semaphore.acquire()
try:
    # critical section
    pass
finally:
    semaphore.release()
```

## Example
```python
import threading
semaphore = threading.Semaphore(3)

def worker():
    semaphore.acquire()
    print("Working...")
    semaphore.release()

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Features
- Controls access to a resource pool
- Useful for limiting concurrency
