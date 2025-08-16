# `shutil` Module 

The **`shutil` (shell utilities)** module provides **high-level file operations** beyond the basic file handling (`open`, `os`, `pathlib`). It deals with **file copying, moving, deletion, directory management, archiving, and disk usage**. It abstracts many shell-level operations into Python functions.

---

## Key Capabilities

* **File Operations**

  * Copying (`copy`, `copy2`, `copyfile`, `copytree`)
  * Moving (`move`)
  * Removal (`rmtree`)
  * Permissions copying (`copymode`, `copystat`)

* **Directory Tree Operations**

  * Copy entire directory trees with metadata
  * Remove directory trees recursively

* **Archiving**

  * Create archives (`make_archive`)
  * Extract archives (`unpack_archive`)
  * Register new archive formats (`register_archive_format`)

* **Disk & Space Management**

  * `disk_usage` for space stats (total, used, free)
  * `chown` for ownership change

* **File Descriptor Operations**

  * `copyfileobj` to copy data from one file-like object to another

* **Temporary Support**

  * Works with temporary directories for staged operations

---

## Commonly Used Functions (Landscape View)

| Category     | Functions                                                                            | Description                         |
| ------------ | ------------------------------------------------------------------------------------ | ----------------------------------- |
| File Copying | `copyfile(src, dst)`, `copy(src, dst)`, `copy2(src, dst)`                            | Copy files with or without metadata |
| File Objects | `copyfileobj(fsrc, fdst, length=16*1024)`                                            | Copy file-like objects in chunks    |
| Directory    | `copytree(src, dst)`                                                                 | Copy directory tree                 |
| Move/Delete  | `move(src, dst)`, `rmtree(path)`                                                     | Move or delete recursively          |
| Metadata     | `copymode(src, dst)`, `copystat(src, dst)`                                           | Copy file permissions, timestamps   |
| Archiving    | `make_archive(base_name, format, root_dir)`, `unpack_archive(filename, extract_dir)` | Create/extract archives             |
| Disk Usage   | `disk_usage(path)`                                                                   | Returns total, used, free bytes     |
| Ownership    | `chown(path, user=None, group=None)`                                                 | Change file owner/group             |

---

## Syntax Highlights

```python
import shutil

# Copy a file
shutil.copy("src.txt", "dest.txt")

# Copy with metadata (timestamps, permissions)
shutil.copy2("src.txt", "dest.txt")

# Copy a directory
shutil.copytree("src_dir", "dest_dir")

# Move a file or directory
shutil.move("src.txt", "backup/src.txt")

# Remove a directory tree
shutil.rmtree("old_dir")

# Create an archive
shutil.make_archive("backup", "zip", "project_dir")

# Extract an archive
shutil.unpack_archive("backup.zip", "restore_dir")

# Get disk usage
total, used, free = shutil.disk_usage("/")

# Copy between file objects
with open("src.txt", "rb") as fsrc, open("dest.txt", "wb") as fdst:
    shutil.copyfileobj(fsrc, fdst)

# Change file ownership (Unix only)
shutil.chown("file.txt", user="username", group="groupname")
```

---

## Usage Scenarios

* Automating **backup and restore**
* Creating and extracting **compressed archives**
* Implementing **deployment scripts**
* File/directory **migration and reorganization**
* Checking **disk availability** before operations
* Handling **cross-platform file operations** at a high level

---
