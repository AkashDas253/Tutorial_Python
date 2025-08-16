# Writing Files in Python 

File writing in Python allows creating new files, modifying existing ones, or appending data. Python provides built-in functions and methods to handle all forms of writing.

---

## File Modes for Writing

* **`'w'`** → Write mode (overwrites file if exists, creates if not).
* **`'a'`** → Append mode (adds data to end of file).
* **`'x'`** → Exclusive creation (creates file, error if exists).
* **`'w+'`** → Write and read mode (overwrites).
* **`'a+'`** → Append and read mode.
* **`'x+'`** → Create, write, and read.

Binary modes can be combined with above (`'wb'`, `'ab'`, etc.) for binary data.

---

## Key Functions & Methods

* **`open(filename, mode, encoding)`** → Open a file for writing.
* **`file.write(string)`** → Write a single string.
* **`file.writelines(list_of_strings)`** → Write multiple lines at once.
* **`file.flush()`** → Force buffer write to disk.
* **`with open(...) as f:`** → Context manager to handle closing automatically.

---

## Syntax & Usage

### Write to a new file

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")   # Writes a single string
    f.write("Overwrites if file already exists.")
```

### Append to a file

```python
with open("output.txt", "a", encoding="utf-8") as f:
    f.write("\nThis line is appended at the end.")
```

### Write multiple lines

```python
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)   # Writes a list of strings
```

### Binary write

```python
data = b"BinaryData\x00\x01"
with open("binaryfile.bin", "wb") as f:
    f.write(data)
```

---

## Best Practices

* Always use **context managers (`with`)** to ensure files close properly.
* Use **encoding (`utf-8`)** for text files.
* Flush or close file explicitly if handling large writes.
* Be cautious with `'w'` mode (overwrites content).

---
