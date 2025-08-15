## Loops in Python

### Overview

Loops allow executing a block of code repeatedly until a certain condition is met. Python supports multiple looping constructs.

---

### Types of Loops

#### **`for` Loop**

* Iterates over a sequence (list, tuple, string, dictionary, range, etc.).
* Syntax:

```python
for variable in sequence:
    # code block
else:
    # optional block executed if loop completes without `break`
```

* Example:

```python
for i in range(3):
    print(i)
else:
    print("Loop finished")
```

---

#### **`while` Loop**

* Executes as long as the condition is `True`.
* Syntax:

```python
while condition:
    # code block
else:
    # optional block executed if loop completes without `break`
```

* Example:

```python
count = 0
while count < 3:
    print(count)
    count += 1
else:
    print("Loop finished")
```

---

### Loop Control Statements

#### **`break`**

* Exits the loop immediately.

```python
for i in range(5):
    if i == 3:
        break
    print(i)
```

#### **`continue`**

* Skips the rest of the code in the current iteration.

```python
for i in range(5):
    if i == 2:
        continue
    print(i)
```

#### **`pass`**

* Placeholder statement that does nothing.

```python
for i in range(5):
    if i == 3:
        pass
    print(i)
```

---

### Nested Loops

* Loops inside other loops.

```python
for i in range(2):
    for j in range(3):
        print(i, j)
```

---

### Infinite Loops

* Loop that runs indefinitely until explicitly broken.

```python
while True:
    print("Press Ctrl+C to stop")
    break
```

---
