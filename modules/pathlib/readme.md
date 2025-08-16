# Pathlib Module

`pathlib` is a standard Python module that provides an **object-oriented interface** for filesystem paths. It makes file and directory operations easier, consistent, and more readable compared to traditional `os` and `os.path` functions.

---

## Pathlib Classes

* **`pathlib.Path`**

  * Abstract class representing system paths.
  * Automatically selects `PosixPath` or `WindowsPath` depending on OS.

* **`pathlib.PosixPath`**

  * Path implementation for Unix-like systems.

* **`pathlib.WindowsPath`**

  * Path implementation for Windows systems.

---

## Creating Paths

```python
from pathlib import Path

# Create a Path object
p = Path("example.txt")

# Absolute path
abs_p = Path("/home/user/docs/file.txt")

# Home directory
home = Path.home()

# Current working directory
cwd = Path.cwd()

# Join paths
joined = Path("folder") / "subfolder" / "file.txt"
```

---

## Path Properties

```python
p = Path("example.txt")

p.name       # 'example.txt' → Full name with extension
p.stem       # 'example' → Name without extension
p.suffix     # '.txt' → Extension only
p.suffixes   # ['.tar', '.gz'] → Multiple extensions
p.parent     # Parent directory
p.parents    # All ancestor directories (iterable)
p.anchor     # Root part ('C:\\', '/')
p.drive      # Drive (Windows only)
p.parts      # Path components as tuple
```

---

## Path Methods

### Existence and Type Checking

```python
p.exists()      # True if path exists
p.is_file()     # True if path is a file
p.is_dir()      # True if path is a directory
p.is_symlink()  # True if path is a symbolic link
```

### Path Conversion

```python
p.resolve()       # Return absolute resolved path
p.absolute()      # Return absolute path
p.as_posix()      # Convert to POSIX-style string
p.as_uri()        # Convert to URI ('file:///...')
```

### Path Modification

```python
p.with_name("newname.txt")   # Change file name
p.with_suffix(".md")         # Change extension
p.relative_to("folder")      # Relative path
p.joinpath("subdir", "file") # Append parts
```

---

## File Operations

### Reading Files

```python
p.read_text(encoding="utf-8")  # Read text content
p.read_bytes()                 # Read binary content
```

### Writing Files

```python
p.write_text("Hello World", encoding="utf-8")  # Write text (overwrite)
p.write_bytes(b"binary data")                  # Write bytes (overwrite)
```

---

## Directory Operations

```python
p.mkdir(parents=False, exist_ok=False)  # Create directory
p.rmdir()                              # Remove empty directory
```

---

## File/Directory Removal and Renaming

```python
p.unlink(missing_ok=False)     # Delete file
p.rename("newname.txt")        # Rename/move
p.replace("newpath.txt")       # Rename (overwrite if exists)
```

---

## Iterating Over Directories

```python
p.iterdir()     # Generator of Path objects in directory
p.glob("*.txt") # Pattern match (non-recursive)
p.rglob("*.py") # Recursive glob
```

---

## File Metadata

```python
stat = p.stat()
stat.st_size   # Size in bytes
stat.st_mtime  # Last modification time
```

---

## Path Comparison

```python
p1 = Path("a.txt")
p2 = Path("./a.txt")

p1 == p2             # Compare equality
p1.samefile(p2)      # True if both refer to same file
p1.is_relative_to(".") # Python 3.9+: check relative
```

---

## Combining with `os` and `shutil`

Even though `pathlib` replaces many `os`/`os.path` functions, you can still combine:

```python
import os
os.listdir(Path.cwd())  # Works fine
```

---

## Common Syntax Summary

```python
from pathlib import Path

# Create
p = Path("file.txt")

# Info
p.name; p.stem; p.suffix; p.parent; p.parts

# Checks
p.exists(); p.is_file(); p.is_dir()

# Convert
p.resolve(); p.as_posix(); p.as_uri()

# Modify
p.with_name("new.txt"); p.with_suffix(".md")

# File ops
p.read_text(); p.write_text("data")
p.read_bytes(); p.write_bytes(b"data")

# Dir ops
p.mkdir(parents=True, exist_ok=True)
p.rmdir()

# Remove / Rename
p.unlink(); p.rename("new.txt"); p.replace("dest.txt")

# Iterate
p.iterdir(); p.glob("*.txt"); p.rglob("*.txt")

# Metadata
stat = p.stat(); stat.st_size; stat.st_mtime
```

---
