## **File Handling in Python**  

File handling in Python allows reading, writing, and manipulating files stored on disk.

---

## **1. Opening a File**  

### **Syntax**
```python
file = open("filename", "mode")
```
| Mode | Description |
|------|------------|
| `'r'` | Read (default) |
| `'w'` | Write (creates/truncates file) |
| `'a'` | Append (adds to file) |
| `'x'` | Create (fails if file exists) |
| `'b'` | Binary mode |
| `'t'` | Text mode (default) |
| `'+'` | Read & write |

---

## **2. Reading a File**  

### **Example: Read Entire File**
```python
file = open("example.txt", "r")
content = file.read()
print(content)
file.close()
```

### **Example: Read Line by Line**
```python
file = open("example.txt", "r")
for line in file:
    print(line.strip())  # Removes newline characters
file.close()
```

### **Example: Read Specific Number of Characters**
```python
file = open("example.txt", "r")
print(file.read(5))  # Reads first 5 characters
file.close()
```

---

## **3. Writing to a File**  

### **Example: Overwrite File**
```python
file = open("example.txt", "w")
file.write("Hello, world!\n")
file.write("Python file handling.")
file.close()
```

### **Example: Append to File**
```python
file = open("example.txt", "a")
file.write("\nAppending new content!")
file.close()
```

---

## **4. Using `with` Statement**  
The `with` statement automatically closes the file.

### **Example: Reading File**
```python
with open("example.txt", "r") as file:
    print(file.read())
```

### **Example: Writing File**
```python
with open("example.txt", "w") as file:
    file.write("Using 'with' statement.")
```

---

## **5. File Methods**  

| Method | Description |
|--------|------------|
| `read(size)` | Reads characters from file |
| `readline()` | Reads one line at a time |
| `readlines()` | Reads all lines as a list |
| `write(text)` | Writes text to file |
| `writelines(list)` | Writes multiple lines |
| `seek(position)` | Moves cursor to position |
| `tell()` | Returns current position |

### **Example: Using `seek()` and `tell()`**
```python
with open("example.txt", "r") as file:
    print(file.read(5))  # Read first 5 characters
    print(file.tell())    # Show current position
    file.seek(0)          # Move cursor to start
    print(file.read(5))   # Read first 5 characters again
```

---

## **6. Working with Binary Files**  
Binary mode is used for non-text files like images and videos.

### **Example: Reading Binary File**
```python
with open("image.jpg", "rb") as file:
    content = file.read()
    print(content[:10])  # Print first 10 bytes
```

### **Example: Writing Binary File**
```python
with open("copy.jpg", "wb") as file:
    file.write(content)
```

---

## **7. Checking If File Exists**  
```python
import os

if os.path.exists("example.txt"):
    print("File exists")
else:
    print("File not found")
```

---

## **8. Deleting a File**  
```python
import os

if os.path.exists("example.txt"):
    os.remove("example.txt")
else:
    print("File does not exist")
```

---

---
---


## File Handling

- File handling is an important part of any web application.
- Python has several functions for creating, reading, updating, and deleting files.

### File Opening and closing

#### Open file:

```python
var = open(filename, mode="rt")
```

- `open(filename, mode="r")`: Opens a file and returns a file object.
  - `filename`: The name of the file to be opened.
  - `mode`: The mode in which the file is opened. Default is `"r"` (read mode).
    - There are four different methods (modes) for opening a file:
        - `"r"` - Read - Default value. Opens a file for reading, error if the file does not exist
        - `"a"` - Append - Opens a file for appending, creates the file if it does not exist
        - `"w"` - Write - Opens a file for writing, creates the file if it does not exist
        - `"x"` - Create - Creates the specified file, returns an error if the file exists

    - In addition, you can specify if the file should be handled as binary or text mode:
        - `"t"` - Text - Default value. Text mode
        - `"b"` - Binary - Binary mode (e.g. images)


#### Closing:

```python
var.close()
```

- `close()`: Closes the file. No parameters.

### With Open (Safe Usage)

#### Syntax:

```python
with open(filename, mode="r") as file_object:
    # Perform file operations
    # No need to explicitly close the file; it is automatically closed when the block is exited.
```

- `filename`: The name of the file to be opened.
- `mode`: The mode in which the file is opened. Default is `"r"` (read mode).
- `file_object`: The file object that you can use to perform file operations.

#### Benefits

- **Automatic Resource Management:** Ensures that the file is properly closed after its suite finishes.
- **Cleaner Code:** Reduces the need for explicit `try...finally` blocks to close the file.
- **Exception Safety:** Even if an exception occurs within the block, the file will still be closed properly.

### File Write and Read:

- `read(size=-1)`: Reads the entire content of the file or up to `size` bytes.
  - `size`: The number of bytes to read. Default is `-1` (read all).

- `readline(size=-1)`: Reads a single line from the file or up to `size` bytes.
  - `size`: The number of bytes to read. Default is `-1` (read the entire line).

- `readlines(hint=-1)`: Reads all the lines of a file and returns them as a list.
  - `hint`: The number of lines to read. Default is `-1` (read all lines).

- `write(string)`: Writes a string to the file.
  - `string`: The string to be written to the file.

- `writelines(lines)`: Writes a list of strings to the file.
  - `lines`: A list of strings to be written to the file.

- `close()`: Closes the file. No parameters.

- `flush()`: Flushes the internal buffer. No parameters.

### File Object Methods

- `fileno()`: Returns the file descriptor. No parameters.

- `isatty()`: Returns `True` if the file is connected to a terminal device. No parameters.

- `readable()`: Returns `True` if the file can be read. No parameters.

- `writable()`: Returns `True` if the file can be written to. No parameters.

- `seekable()`: Returns `True` if the file supports random access. No parameters.