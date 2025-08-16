# `os` Module in Python

### Overview

* Provides a **portable way** to interact with the operating system.
* Handles file operations, process management, environment variables, and directory navigation.
* Abstracts system-specific functionality to work across platforms (Windows, Linux, macOS).

---

### Key Features

#### File & Directory Operations

* **`os.getcwd()`** → Get current working directory.
* **`os.chdir(path)`** → Change current directory.
* **`os.listdir(path='.')`** → List files and directories.
* **`os.mkdir(path, mode=0o777)`** → Create directory.
* **`os.makedirs(path, exist_ok=False)`** → Create nested directories.
* **`os.remove(path)`** → Delete file.
* **`os.rmdir(path)`** → Remove empty directory.
* **`os.removedirs(path)`** → Remove nested directories.
* **`os.rename(src, dst)`** → Rename file or directory.
* **`os.replace(src, dst)`** → Rename with overwrite support.

---

#### File Metadata & Info

* **`os.stat(path)`** → Get file metadata (size, permissions, etc.).
* **`os.path`** → Submodule for path manipulation:

  * `os.path.join(a, b)` → Join paths.
  * `os.path.exists(path)` → Check if path exists.
  * `os.path.isfile(path)` → Check if path is a file.
  * `os.path.isdir(path)` → Check if path is a directory.
  * `os.path.abspath(path)` → Absolute path.
  * `os.path.basename(path)` → File name.
  * `os.path.dirname(path)` → Directory name.
  * `os.path.splitext(path)` → Split extension.

---

#### Environment Variables

* **`os.environ`** → Dictionary of environment variables.
* **`os.getenv(key, default=None)`** → Get environment variable.
* **`os.putenv(key, value)`** → Set variable (less recommended).
* **`os.unsetenv(key)`** → Remove variable.

---

#### Process Management

* **`os.system(command)`** → Run shell command.
* **`os.popen(command)`** → Execute command, return file-like object.
* **`os.execvp(prog, args)`** → Replace current process with new one.
* **`os.getpid()`** → Get current process ID.
* **`os.getppid()`** → Get parent process ID.
* **`os.fork()`** → Fork process (Unix only).
* **`os._exit(code)`** → Exit immediately.

---

#### Permissions & Ownership

* **`os.chmod(path, mode)`** → Change file permissions.
* **`os.chown(path, uid, gid)`** → Change ownership.
* **`os.umask(mask)`** → Set default file mode creation mask.

---

#### Randomness & File Descriptors

* **`os.urandom(n)`** → Generate `n` random bytes.
* **`os.open(path, flags, mode=0o777)`** → Low-level file open.
* **`os.read(fd, n)`**, **`os.write(fd, str)`** → File descriptor read/write.
* **`os.close(fd)`** → Close descriptor.

---

#### System Information

* **`os.name`** → OS name (`posix`, `nt`).
* **`os.uname()`** → System info (Unix only).
* **`os.cpu_count()`** → Number of CPUs.
* **`os.getlogin()`** → User logged in.

---

### Example

```python
import os

print("CWD:", os.getcwd())          # Current directory
os.mkdir("test_dir")                # Create folder
print(os.listdir("."))              # List contents
print("HOME:", os.getenv("HOME"))   # Get env variable
```

---
