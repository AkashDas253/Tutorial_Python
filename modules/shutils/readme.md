# Shutil Module

`shutil` (Shell Utilities) is a Python standard library module for **high-level file operations** like copying, moving, archiving, and filesystem management.
It complements `os` and `pathlib`, which focus more on low-level path and file handling.

---

## File Operations

### Copying Files

```python
import shutil

shutil.copy(src, dst)  
# Copy file from src → dst (permissions may change)

shutil.copy2(src, dst)  
# Copy file preserving metadata (timestamps, permissions)

shutil.copyfile(src, dst)  
# Copy contents only (no metadata, dst must be a file path)

shutil.copymode(src, dst)  
# Copy permission bits only

shutil.copystat(src, dst)  
# Copy all metadata (permissions, timestamps, flags)
```

---

### Moving and Renaming

```python
shutil.move(src, dst)  
# Move file/dir, similar to Unix mv
```

---

### Deleting

```python
shutil.rmtree(path, ignore_errors=False, onerror=None)  
# Remove directory tree (recursively)

shutil.which(cmd, mode=os.F_OK | os.X_OK, path=None)  
# Locate command in PATH (like Unix 'which')
```

---

## Directory and Tree Operations

```python
shutil.copytree(src, dst, symlinks=False, ignore=None, 
                copy_function=shutil.copy2, dirs_exist_ok=False)  
# Recursively copy directory tree

shutil.ignore_patterns(*patterns)  
# Helper to exclude files during copytree
```

---

## Archiving Operations

```python
shutil.make_archive(base_name, format, root_dir=None, base_dir=None, 
                    verbose=0, dry_run=0, owner=None, group=None, logger=None)  
# Create archive: format = 'zip', 'tar', 'gztar', 'bztar', 'xztar'

shutil.unpack_archive(filename, extract_dir=None, format=None)  
# Extract archive

shutil.get_archive_formats()  
# List supported archive formats

shutil.register_archive_format(name, function, extra_args=None, description='')  
# Register new archive format

shutil.unregister_archive_format(name)  
# Remove archive format
```

---

## Disk Usage and Space Management

```python
shutil.disk_usage(path)  
# Returns (total, used, free) in bytes
```

---

## Temporary File Handling

```python
shutil.chown(path, user=None, group=None)  
# Change owner (Unix only)

shutil.copymode(src, dst)  
# Copy only permissions
```

---

## File Descriptor Operations (Python 3.8+)

```python
shutil.copyfileobj(fsrc, fdst, length=16*1024)  
# Copy data between file objects (streams)
```

---

## High-level File Operations Summary

| Function          | Purpose                          |
| ----------------- | -------------------------------- |
| `copy`            | Copy file (may change metadata)  |
| `copy2`           | Copy file with metadata          |
| `copyfile`        | Copy only contents               |
| `copymode`        | Copy permissions                 |
| `copystat`        | Copy full metadata               |
| `move`            | Move or rename                   |
| `rmtree`          | Delete directory tree            |
| `copytree`        | Copy entire directory tree       |
| `ignore_patterns` | Helper for excluding in copytree |
| `make_archive`    | Create archive                   |
| `unpack_archive`  | Extract archive                  |
| `disk_usage`      | Get total/used/free space        |
| `which`           | Find command in PATH             |
| `copyfileobj`     | Copy between file objects        |

---

## Common Syntax Examples

```python
import shutil

# Copying
shutil.copy("a.txt", "b.txt")
shutil.copy2("a.txt", "b.txt")

# Moving
shutil.move("a.txt", "folder/")

# Deleting
shutil.rmtree("old_folder")

# Directory copy
shutil.copytree("src_dir", "dst_dir", dirs_exist_ok=True)

# Archiving
shutil.make_archive("backup", "zip", "project_folder")
shutil.unpack_archive("backup.zip", "extracted")

# Disk usage
total, used, free = shutil.disk_usage("/")

# Locate command
print(shutil.which("python3"))
```

---
