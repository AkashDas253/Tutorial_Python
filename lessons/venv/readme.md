## `venv` – Built-in Python Environment Manager

---

### What is `venv`?

`venv` is a **built-in module** (Python 3.3+) used to **create lightweight, isolated Python environments**. It isolates dependencies for individual projects to avoid conflicts.

* Ships with Python – no installation needed
* Simpler alternative to `virtualenv` (for Python 3 only)

---

### Create a Virtual Environment

```bash
python -m venv <env_name>              # Default interpreter
python3.10 -m venv <env_name>          # Specific version
```

This creates a directory named `<env_name>` containing the isolated environment.

---

### Activate the Environment

| OS                   | Command                           |
| -------------------- | --------------------------------- |
| Linux/macOS          | `source <env_name>/bin/activate`  |
| Windows (cmd)        | `<env_name>\Scripts\activate.bat` |
| Windows (PowerShell) | `<env_name>\Scripts\Activate.ps1` |

After activation, your shell prompt changes to show the environment name.

---

### Deactivate

```bash
deactivate
```

Returns to the global Python environment.

---

### Optional Flags

| Flag                     | Description                                 |
| ------------------------ | ------------------------------------------- |
| `--system-site-packages` | Give access to global site packages         |
| `--clear`                | Clear environment directory before creating |
| `--upgrade`              | Upgrade environment in place if it exists   |
| `--prompt <name>`        | Customize prompt shown in shell             |

---

### Directory Structure

| Path                 | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `bin/` or `Scripts/` | Contains executables like `python`, `pip`, and `activate` |
| `lib/`               | Installed libraries                                       |
| `pyvenv.cfg`         | Configuration file for environment                        |
| `include/`           | Header files for C extensions                             |

---

### Installing Packages Inside the Env

Once activated:

```bash
pip install <package>
pip freeze > requirements.txt
```

---

### `venv` vs `virtualenv`

| Feature            | `venv`    | `virtualenv`              |
| ------------------ | --------- | ------------------------- |
| Built-in           | ✅ Yes     | ❌ Needs pip install       |
| Python 2 support   | ❌ No      | ✅ Yes                     |
| Custom interpreter | ❌ Limited | ✅ Full support            |
| Speed              | 🟢 Fast   | 🟢 Fast (slightly faster) |

---

### Remove an Environment

Just delete the environment directory:

```bash
rm -rf <env_name>
```

---
