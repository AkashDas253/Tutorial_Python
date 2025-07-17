## Input/Output in Python 

### Definition

Input and output (I/O) in Python refers to **reading data into a program (input)** and **sending data from the program to the outside world (output)**. Python supports multiple methods of I/O for various mediums like keyboard, screen, file, memory, and network.

---

### Categories of I/O in Python

* **Standard I/O**

  * **Input:** From keyboard using `input()` or low-level `sys.stdin`
  * **Output:** To screen using `print()` or low-level `sys.stdout`

* **File I/O**

  * Persistent storage interaction through `open()`, file objects, file modes, and buffering

* **Binary vs Text I/O**

  * **Text Mode:** Handles string data; includes line endings translation
  * **Binary Mode:** Handles bytes; no transformation

* **Formatted I/O**

  * String formatting via `f-strings`, `format()`, or `%` operator

* **Buffered vs Unbuffered I/O**

  * **Buffered I/O:** Stores data temporarily in memory to optimize speed
  * **Unbuffered I/O:** Direct I/O, often slower but more immediate

---

### Streams in Python

| Type          | Description                               |
| ------------- | ----------------------------------------- |
| Text Stream   | Handles characters; decodes bytes to text |
| Binary Stream | Handles raw byte data                     |
| Buffered      | High-level wrapper for performance        |
| Raw           | Unbuffered, lowest-level byte interface   |

---

### Input Sources

* **Standard Input (Keyboard)**

  * `input()` → always returns string
  * `sys.stdin` for more control (line-buffered)

* **File Input**

  * Reading from `.txt`, `.csv`, `.json`, etc.
  * Can be sequential or random-access

* **In-memory Input**

  * Using `io.StringIO` or `io.BytesIO` to mimic file I/O in memory

* **Network Input**

  * Via sockets or APIs (requires additional modules)

---

### Output Targets

* **Standard Output (Screen)**

  * `print()`, `sys.stdout.write()`

* **File Output**

  * Write/append data to files
  * File modes like `'w'`, `'a'`, `'r+'`, `'x'`

* **In-memory Output**

  * `io.StringIO`, `io.BytesIO`

* **Network Output**

  * Data written to sockets or remote servers

---

### Modes of File I/O

| Mode   | Description                      |
| ------ | -------------------------------- |
| `'r'`  | Read (text)                      |
| `'rb'` | Read (binary)                    |
| `'w'`  | Write (overwrite, text)          |
| `'wb'` | Write (overwrite, binary)        |
| `'a'`  | Append (text)                    |
| `'ab'` | Append (binary)                  |
| `'x'`  | Create new file, fail if exists  |
| `'r+'` | Read & write (text, no truncate) |

---

### I/O Abstractions

* **File object**: High-level interface to file data
* **Stream**: Abstract layer over input/output
* **Buffer**: Temporary memory for optimizing I/O

---

### Memory I/O (In-Memory Buffers)

* Useful for **testing** and **temporary storage**
* `io.StringIO`: for text
* `io.BytesIO`: for binary

---

### Error Handling in I/O

* Common Exceptions:

  * `FileNotFoundError`
  * `PermissionError`
  * `IsADirectoryError`
  * `UnicodeDecodeError` / `UnicodeEncodeError`

* Best Practice:

  * Always use `try...except` or context managers (`with`)

---

### Context Manager (`with` Statement)

* Automatically handles:

  * File opening
  * Resource management
  * Closing file (even on error)

---

### Unicode Handling

* Python 3 supports Unicode natively
* Ensure correct encoding/decoding (`utf-8`, `ascii`, `latin-1`, etc.)
* Always specify `encoding` explicitly for cross-platform compatibility

---

### Data Serialization I/O

* For writing/reading structured data:

  * **JSON:** via `json` module
  * **Pickle:** via `pickle` (Python object serialization)
  * **CSV/XML/YAML:** via respective modules
* These enable structured input/output beyond plain text or binary

---

### Buffered Layers (File I/O Stack)

| Layer      | Class                  | Description              |
| ---------- | ---------------------- | ------------------------ |
| High-level | `TextIOWrapper`        | Text encoding/decoding   |
| Buffered   | `BufferedReader`, etc. | Performance optimization |
| Raw        | `FileIO`               | Basic read/write from OS |

---

### Advanced Topics

* **Redirection of input/output**

  * Using `sys.stdin`, `sys.stdout`, and `sys.stderr` redirection
* **Non-blocking I/O**

  * Via `select`, `asyncio`, or threading for concurrent operations
* **File Descriptor I/O**

  * Low-level interaction with OS descriptors via `os` module
* **Logging Output**

  * Using `logging` module for structured output instead of `print()`

---
