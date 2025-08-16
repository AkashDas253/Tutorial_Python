# File Reading in Python

File reading is a core part of file handling in Python. It involves opening a file, extracting data in various formats (text, binary, line-wise, chunk-wise), and managing resources efficiently.

---

## Key Concepts in File Reading

* **File Modes for Reading**

  * `'r'`: Read (default, text mode).
  * `'rb'`: Read in binary mode.
  * `'r+'`: Read and write (file must exist).
  * `'rb+'`: Binary read and write.

* **Opening a File**

  * `open(file, mode, buffering, encoding, errors, newline, closefd, opener)`

    * `file`: Path to the file.
    * `mode`: File access mode.
    * `encoding`: For text mode (e.g., `'utf-8'`).
    * `buffering`: Buffer policy (`-1` default, `0` no buffer, `1` line-buffered).
    * `errors`: Error handling (`'strict'`, `'ignore'`, `'replace'`).
    * `newline`: Controls newline translation.
    * `closefd`: If `False`, keeps file descriptor open.
    * `opener`: Custom opener.

* **Reading Methods**

  * `file.read(size=-1)` → Reads entire file or up to `size` characters/bytes.
  * `file.readline(size=-1)` → Reads one line (optionally up to `size` chars).
  * `file.readlines(hint=-1)` → Reads all lines into a list.
  * Iterating with `for line in file` → Efficient line-by-line reading.
  * `next(file)` → Reads next line manually.

* **Binary Reading**

  * Returns `bytes` objects.
  * Useful for images, executables, serialized data.

* **Efficient Reading**

  * Use `with` statement for automatic resource cleanup.
  * Iterate directly over file objects for large files.
  * Use chunk reading for memory efficiency (`file.read(1024)`).

* **File Pointer Movement**

  * `file.tell()` → Current cursor position.
  * `file.seek(offset, whence=0)`

    * `whence=0`: from start (default).
    * `whence=1`: from current position.
    * `whence=2`: from end.

* **Closing the File**

  * `file.close()` → Manually closes file if not using `with`.

---

## Syntax Examples

### Basic File Reading

```python
# Open and read entire file
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

### Reading Line by Line

```python
# Line iteration
with open("example.txt", "r") as f:
    for line in f:
        print(line.strip())
```

### Reading Specific Number of Characters

```python
with open("example.txt", "r") as f:
    part = f.read(50)  # Read first 50 chars
```

### Reading Binary Files

```python
with open("image.png", "rb") as f:
    data = f.read()
```

### File Pointer Handling

```python
with open("example.txt", "r") as f:
    f.seek(10)         # Move pointer to 10th byte
    print(f.read(20))  # Read 20 characters
    print(f.tell())    # Get current position
```

### Manual Opening and Closing

```python
f = open("example.txt", "r")
try:
    print(f.read())
finally:
    f.close()
```

---
