## **Multithreading in Python**  

**Multithreading** allows a program to run multiple tasks (threads) concurrently, improving performance for I/O-bound tasks. Python’s `threading` module provides tools for creating and managing threads.

---

## **1. Creating a Thread**  

### **Syntax**
```python
import threading

def function_name():
    # Task to perform

thread = threading.Thread(target=function_name)
thread.start()
```

### **Example**
```python
import threading

def print_hello():
    print("Hello from thread!")

thread = threading.Thread(target=print_hello)
thread.start()
```

---

## **2. Running Multiple Threads**
```python
import threading

def task(n):
    print(f"Task {n} running")

threads = []
for i in range(5):  # Create 5 threads
    thread = threading.Thread(target=task, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()  # Wait for all threads to complete
```
📌 **`join()` ensures the main program waits for threads to finish.**

---

## **3. Using `Thread` Class**
```python
import threading

class MyThread(threading.Thread):
    def run(self):
        print("Thread running")

t = MyThread()
t.start()
t.join()
```

---

## **4. Thread Synchronization with Locks**  
**Locks prevent multiple threads from accessing shared resources simultaneously.**
```python
import threading

lock = threading.Lock()

def task(n):
    with lock:  # Ensures only one thread accesses this section at a time
        print(f"Task {n} executing")

threads = [threading.Thread(target=task, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## **5. Threading with `Queue` for Safe Communication**
```python
import threading
import queue

q = queue.Queue()

def worker():
    while not q.empty():
        item = q.get()
        print(f"Processing {item}")
        q.task_done()

for i in range(5):
    q.put(i)

threads = [threading.Thread(target=worker) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## **6. Daemon Threads**  
Daemon threads run in the background and exit when the main program terminates.
```python
import threading
import time

def background_task():
    while True:
        print("Running...")
        time.sleep(1)

t = threading.Thread(target=background_task, daemon=True)
t.start()
time.sleep(3)  # Main program ends, daemon thread stops
```

---
