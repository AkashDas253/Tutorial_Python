## File Handling – OS Module Support

The **`os` module** provides low-level file handling operations that complement Python’s built-in file objects. While the `open()` function and context managers cover high-level usage, the `os` module offers **system-level control** over files and file descriptors.

---

### Key Points

* Works at a **lower abstraction layer** than `open()`.
* Operates with **file descriptors (integers)** instead of file objects.
* Useful for scenarios where fine-grained control over files is needed.
* Integrates closely with the **operating system’s API**.

---

### Commonly Used Functions in File Handling

| Function                           | Purpose                                     |
| ---------------------------------- | ------------------------------------------- |
| `os.open(path, flags, mode=0o777)` | Open a file and return a file descriptor    |
| `os.read(fd, n)`                   | Read `n` bytes from file descriptor         |
| `os.write(fd, string)`             | Write bytes to file descriptor              |
| `os.close(fd)`                     | Close file descriptor                       |
| `os.lseek(fd, pos, how)`           | Move file pointer to new position           |
| `os.remove(path)`                  | Delete a file                               |
| `os.rename(src, dst)`              | Rename a file                               |
| `os.stat(path)`                    | Get file metadata (size, permissions, etc.) |
| `os.fstat(fd)`                     | Get metadata for an open file descriptor    |
| `os.path.exists(path)`             | Check file existence                        |

---

### Syntax Examples

```python
import os

# Open file (O_RDWR = read/write)
fd = os.open("example.txt", os.O_RDWR | os.O_CREAT)

# Write to file
os.write(fd, b"Hello OS Module")

# Move file pointer to start
os.lseek(fd, 0, os.SEEK_SET)

# Read content
content = os.read(fd, 20)
print(content.decode())

# Close file descriptor
os.close(fd)
```

---

### OS File Access Modes (Flags)

* `os.O_RDONLY` → Read only
* `os.O_WRONLY` → Write only
* `os.O_RDWR` → Read & write
* `os.O_CREAT` → Create if not exists
* `os.O_EXCL` → Error if file exists (used with `O_CREAT`)
* `os.O_TRUNC` → Truncate file to zero length
* `os.O_APPEND` → Append at the end of file

---

### Advantages

* Gives **fine-grained, system-level control**.
* Better suited for **low-level operations** and **performance-critical systems**.
* Works consistently across different platforms (with minor differences).

---

### Usage Scenarios

* When working with **raw file descriptors** (instead of Python file objects).
* Building **custom file handling systems**.
* Interfacing with **legacy code** or **C libraries**.
* Managing files in **system programming** or **low-level utilities**.

---
