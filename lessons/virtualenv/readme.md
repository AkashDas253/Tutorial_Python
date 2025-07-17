## `virtualenv` – Python Environment Isolation Tool

---

### What is `virtualenv`?

`virtualenv` is a **tool to create isolated Python environments**, each with its own **interpreter, libraries, and scripts**, independent from the system Python.

* Useful for projects needing different dependencies or Python versions
* Works with both Python 2 and 3

---

### Installation

```bash
pip install virtualenv
```

Check version:

```bash
virtualenv --version
```

---

### Creating a Virtual Environment

```bash
virtualenv <env_name>             # Default interpreter
virtualenv -p python3.9 <env_name>  # Specific Python version
```

---

### Activating the Environment

| OS                   | Command                           |
| -------------------- | --------------------------------- |
| Linux/macOS          | `source <env_name>/bin/activate`  |
| Windows (cmd)        | `<env_name>\Scripts\activate.bat` |
| Windows (PowerShell) | `<env_name>\Scripts\Activate.ps1` |

---

### Deactivating

```bash
deactivate
```

---

### Deleting an Environment

* Just delete the folder:

  ```bash
  rm -rf <env_name>
  ```

---

### Options

| Option                   | Description                                 |
| ------------------------ | ------------------------------------------- |
| `-p /usr/bin/pythonX`    | Choose Python interpreter                   |
| `--clear`                | Clear existing packages in env              |
| `--system-site-packages` | Give access to global site packages         |
| `--no-site-packages`     | (Default) Isolate from global site packages |

---

### Structure of a `virtualenv`

| Folder/File          | Description                         |
| -------------------- | ----------------------------------- |
| `bin/` or `Scripts/` | Executables (activate, python, pip) |
| `lib/`               | Installed packages                  |
| `pyvenv.cfg`         | Config file for environment         |
| `include/`           | C headers (if needed)               |

---

### `virtualenv` vs `venv`

| Feature               | `virtualenv`                   | `venv`              |
| --------------------- | ------------------------------ | ------------------- |
| Comes with Python     | ❌ No (install via pip)         | ✅ Yes (Python 3.3+) |
| Cross-version support | ✅ Yes (e.g., Python 2.7, 3.4+) | ❌ Python 3+ only    |
| Speed/features        | ✅ Faster, richer               | Simpler             |

---

### Usage with `pip`

Once activated:

```bash
pip install <package>          # Installed inside the virtualenv
pip freeze > requirements.txt  # Export project deps
```

---

Would you like an illustrated workflow comparing `virtualenv`, `venv`, and `conda` environment management?
