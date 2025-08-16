## File Iteration in Python

File iteration is the process of reading a file line by line without explicitly loading the entire file into memory. Python file objects are **iterators**, meaning they support iteration with loops.

---

### Key Points

* Uses the file object as an **iterator**.
* Reads one line at a time, making it **memory-efficient** for large files.
* Iteration stops automatically at EOF (end of file).
* Supports both `for` loops and `next()` function.
* Each line returned includes the newline character `\n` unless stripped.

---

### Syntax

```python
# Using for loop
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())   # strip removes newline characters

# Using next()
with open("example.txt", "r") as file:
    print(next(file))  # First line
    print(next(file))  # Second line
```

---

### Methods and Properties in Iteration

* `file.__iter__()` → Returns the file object itself (iterator).
* `file.__next__()` → Returns next line, raises `StopIteration` at EOF.
* Works seamlessly with **`for` loop** since it calls these methods internally.

---

### Usage Scenarios

* Processing large log files line by line.
* Streaming data from text/CSV files.
* Avoiding memory overload with massive files.

---
