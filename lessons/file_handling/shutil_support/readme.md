# File Handling – `shutil` Module

### Overview

The `shutil` module provides high-level operations on files and directories, building on top of `os` and `pathlib`. It is commonly used for file copying, moving, archiving, and disk usage management. It complements low-level operations by simplifying common tasks.

---

### Key Features

* File and Directory Operations
* Copying and Moving
* Removing Trees
* Archiving
* Disk Usage Information
* File System Metadata Handling

---

### Functions

#### File Copying

* `shutil.copy(src, dst)`
  Copies file content + permissions.
* `shutil.copy2(src, dst)`
  Same as `copy()` but preserves metadata (timestamps, flags).
* `shutil.copyfile(src, dst)`
  Copies only file content (no metadata).
* `shutil.copyfileobj(fsrc, fdst, length=16*1024)`
  Copies data between file objects.

#### File and Directory Moving

* `shutil.move(src, dst)`
  Moves file/directory to another location (cross-filesystem supported).

#### Removing

* `shutil.rmtree(path)`
  Recursively deletes a directory tree.

#### Archiving

* `shutil.make_archive(base_name, format, root_dir)`
  Creates archive (`zip`, `tar`, `gztar`, etc.).
* `shutil.unpack_archive(filename, extract_dir=None, format=None)`
  Extracts archive contents.

#### Disk Usage

* `shutil.disk_usage(path)`
  Returns named tuple `(total, used, free)` in bytes.

#### Temporary Files & Directories

* `shutil.get_archive_formats()`
  Returns supported archive formats.
* `shutil.get_unpack_formats()`
  Returns supported unpack formats.

#### File System Metadata

* `shutil.chown(path, user=None, group=None)`
  Changes owner and group of path.
* `shutil.copymode(src, dst)`
  Copies file permission mode.
* `shutil.copystat(src, dst)`
  Copies all metadata.

---

### Syntax Examples

```python
import shutil

# Copy file
shutil.copy("source.txt", "destination.txt")

# Copy with metadata
shutil.copy2("source.txt", "destination.txt")

# Move file
shutil.move("file.txt", "backup/file.txt")

# Remove directory tree
shutil.rmtree("old_logs")

# Create archive
shutil.make_archive("backup", "zip", "project_folder")

# Extract archive
shutil.unpack_archive("backup.zip", "restore_folder")

# Disk usage
usage = shutil.disk_usage("/")
print(usage.total, usage.used, usage.free)

# Change ownership (Unix only)
shutil.chown("script.py", user="admin", group="dev")
```

---
