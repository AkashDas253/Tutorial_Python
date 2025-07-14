## **Multithreading vs. Multiprocessing in Python**  

Both **multithreading** and **multiprocessing** allow concurrent execution in Python, but they serve different purposes:  

- **Multithreading** is useful for **I/O-bound tasks** (e.g., network requests, file I/O, database queries).  
- **Multiprocessing** is ideal for **CPU-bound tasks** (e.g., mathematical computations, image processing).  

---

## **1. Multithreading in Python**  

Python’s `threading` module enables multithreading, but due to the **Global Interpreter Lock (GIL)**, only one thread executes at a time.  

### **Syntax: Creating a Thread**  
```python
import threading

def task():
    print("Task running")

thread = threading.Thread(target=task)
thread.start()
thread.join()  # Ensures the main thread waits for completion
```

### **Example: Running Multiple Threads**  
```python
import threading

def worker(n):
    print(f"Thread {n} is working")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```
📌 **Useful for I/O-bound tasks, but not CPU-bound tasks due to GIL.**

---

## **2. Multiprocessing in Python**  

The `multiprocessing` module allows parallel execution by creating separate processes, each with its own memory space.

### **Syntax: Creating a Process**  
```python
from multiprocessing import Process

def task():
    print("Process running")

process = Process(target=task)
process.start()
process.join()
```

### **Example: Running Multiple Processes**  
```python
from multiprocessing import Process

def worker(n):
    print(f"Process {n} is working")

processes = []
for i in range(3):
    p = Process(target=worker, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()
```
📌 **True parallel execution as each process runs in a separate memory space.**

---

## **3. Key Differences Between Multithreading and Multiprocessing**

| Feature | Multithreading | Multiprocessing |
|---------|--------------|---------------|
| **Execution** | Concurrent (one thread at a time due to GIL) | Parallel (separate processes) |
| **Best for** | I/O-bound tasks | CPU-bound tasks |
| **Memory Usage** | Shared memory | Separate memory space for each process |
| **Performance** | Limited by GIL for CPU tasks | Efficient for CPU-heavy tasks |
| **Overhead** | Low (threads share memory) | Higher (each process has its own memory) |
| **Example Use Cases** | Web scraping, I/O operations | Image processing, ML model training |

---

## **4. When to Use Multithreading vs. Multiprocessing?**  

- **Use Multithreading if:**  
  - Your tasks involve I/O operations (e.g., reading files, network requests).  
  - You need lightweight concurrency without extra memory overhead.  

- **Use Multiprocessing if:**  
  - Your tasks involve heavy CPU computations.  
  - You need true parallel execution, bypassing GIL.  

---

## **5. Using Both Together (Hybrid Approach)**  
Sometimes, combining **multithreading** and **multiprocessing** can be beneficial.  

```python
import threading
from multiprocessing import Process

def cpu_task(n):
    print(f"CPU-bound task {n}")

def io_task(n):
    print(f"I/O-bound task {n}")

if __name__ == "__main__":
    p = Process(target=cpu_task, args=(1,))
    t = threading.Thread(target=io_task, args=(2,))
    
    p.start()
    t.start()
    
    p.join()
    t.join()
```
📌 **CPU-bound tasks run in a process, while I/O-bound tasks run in a thread.**

---
