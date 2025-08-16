# Thread Synchronization: Lock

A `Lock` is the simplest synchronization primitive in Python's `threading` module. It is used to ensure that only one thread can access a resource at a time.

## Syntax
```python
import threading
lock = threading.Lock()
```

## Usage
```python
lock.acquire()
try:
    # critical section
    pass
finally:
    lock.release()
```

## Example
```python
import threading
counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(1000):
        lock.acquire()
        counter += 1
        lock.release()

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(counter)
```

## Features
- Prevents race conditions
- Only one thread can hold the lock at a time
