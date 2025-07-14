## **Multiprocessing in Python**  

**Multiprocessing** allows Python programs to execute multiple processes in parallel, making it ideal for CPU-bound tasks. Unlike **multithreading**, multiprocessing **bypasses Python's Global Interpreter Lock (GIL)**, enabling true parallel execution. The `multiprocessing` module provides tools to create and manage processes.

---

## **1. Creating a Process**  

### **Syntax**  
```python
from multiprocessing import Process

def function_name():
    # Task to perform

process = Process(target=function_name)
process.start()
process.join()
```

### **Example**
```python
from multiprocessing import Process

def print_hello():
    print("Hello from process!")

process = Process(target=print_hello)
process.start()
process.join()  # Ensures the main program waits for the process to complete
```

---

## **2. Running Multiple Processes**  
```python
from multiprocessing import Process

def task(n):
    print(f"Task {n} running")

processes = []
for i in range(5):  # Create 5 processes
    p = Process(target=task, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()  # Wait for all processes to complete
```
📌 **Each process runs independently in a separate memory space.**

---

## **3. Using `Process` Class**
```python
from multiprocessing import Process

class MyProcess(Process):
    def run(self):
        print("Process running")

p = MyProcess()
p.start()
p.join()
```

---

## **4. Inter-Process Communication (IPC) with Queue**
Processes don’t share memory, so we use `Queue` for communication.
```python
from multiprocessing import Process, Queue

def worker(q):
    q.put("Data from process")

q = Queue()
p = Process(target=worker, args=(q,))
p.start()
p.join()

print(q.get())  # Output: Data from process
```

---

## **5. Using `Pool` for Parallel Execution**
```python
from multiprocessing import Pool

def square(n):
    return n * n

with Pool(processes=4) as pool:
    result = pool.map(square, [1, 2, 3, 4, 5])
    print(result)  # Output: [1, 4, 9, 16, 25]
```
📌 **`Pool` automatically distributes tasks across multiple processes.**

---

## **6. Shared Memory with `Value` and `Array`**
```python
from multiprocessing import Process, Value

def increment(val):
    val.value += 1

num = Value("i", 0)  # Shared integer
p = Process(target=increment, args=(num,))
p.start()
p.join()

print(num.value)  # Output: 1
```

---

## **7. Using `Lock` for Synchronization**  
Locks prevent race conditions in shared resources.
```python
from multiprocessing import Process, Lock

lock = Lock()

def task(n):
    with lock:  # Ensures only one process accesses this section at a time
        print(f"Task {n} executing")

processes = [Process(target=task, args=(i,)) for i in range(3)]
for p in processes:
    p.start()
for p in processes:
    p.join()
```

---

## **8. Daemon Processes**  
Daemon processes run in the background and terminate when the main process exits.
```python
from multiprocessing import Process
import time

def background_task():
    while True:
        print("Running...")
        time.sleep(1)

p = Process(target=background_task, daemon=True)
p.start()
time.sleep(3)  # Main process ends, daemon process stops
```

---
