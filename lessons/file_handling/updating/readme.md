# File Updating in Python 

Updating a file means **modifying its contents without fully discarding existing data**. Python allows this using **read + write modes** and specific methods.

---

## File Modes for Updating

* **`r+`** → Read & write (does not create file, error if missing).
* **`w+`** → Write & read (creates file if missing, overwrites if exists).
* **`a+`** → Append & read (creates file if missing, adds data to end).
* **`rb+` / `wb+` / `ab+`** → Binary equivalents.

---

## Key Functions & Methods

* **`file.read(size)`** → Read existing data.
* **`file.write(string)`** → Overwrite or add new data.
* **`file.seek(offset, whence)`** → Move cursor to update at specific position.

  * `whence=0` → From beginning (default).
  * `whence=1` → From current position.
  * `whence=2` → From end of file.
* **`file.truncate(size)`** → Shrink or expand file to given size.

---

## Syntax & Usage

### Update at the beginning

```python
with open("data.txt", "r+", encoding="utf-8") as f:
    content = f.read()
    f.seek(0)  
    f.write("Updated first line\n")
    f.write(content)  # Keep old content
```

### Update at a specific position

```python
with open("data.txt", "r+", encoding="utf-8") as f:
    f.seek(10)  
    f.write("INSERTED")
```

### Append with read access

```python
with open("data.txt", "a+", encoding="utf-8") as f:
    f.write("\nAppended line")
    f.seek(0)  
    print(f.read())  # Read entire updated file
```

### Truncate content after update

```python
with open("data.txt", "r+", encoding="utf-8") as f:
    f.write("Keep only this text")
    f.truncate()
```

### Update binary file

```python
with open("binary.bin", "rb+") as f:
    f.seek(2)  
    f.write(b"\xFF\xFF")
```

---

## Best Practices

* Use `r+` for safe updating (avoids overwrite).
* Use `seek()` carefully to position updates.
* Always combine reading + writing logic to prevent data loss.
* Use `truncate()` to remove unwanted content after overwriting.

---
