# Thread Synchronization: RLock

An `RLock` (reentrant lock) allows a thread to acquire the same lock multiple times. Useful for recursive code.

## Syntax
```python
import threading
rlock = threading.RLock()
```

## Usage
```python
rlock.acquire()
try:
    # critical section
    pass
finally:
    rlock.release()
```

## Example
```python
import threading
rlock = threading.RLock()

def recursive_function(n):
    rlock.acquire()
    if n > 0:
        recursive_function(n-1)
    rlock.release()

recursive_function(5)
```

## Features
- Allows the same thread to acquire the lock multiple times
- Useful for recursive functions
