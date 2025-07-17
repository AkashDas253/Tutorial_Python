## `pip` – Python’s Default Package Manager

---

### What is `pip`?

`pip` is the **default package installer** for Python. It installs Python packages from the [Python Package Index (PyPI)](https://pypi.org).

---

### Installation

* Comes pre-installed with Python 3.4+
* Check version:

  ```bash
  pip --version
  ```
* Install (if missing):

  ```bash
  python -m ensurepip
  ```

---

### Basic Syntax

```bash
pip install [options] <package> [package ...]
```

---

### Common Commands

| Command                           | Description                                         |
| --------------------------------- | --------------------------------------------------- |
| `pip install <pkg>`               | Install latest version                              |
| `pip install <pkg>==1.2.3`        | Install specific version                            |
| `pip install <pkg>~=1.2.0`        | Install compatible version                          |
| `pip install -r requirements.txt` | Install packages from file                          |
| `pip uninstall <pkg>`             | Remove a package                                    |
| `pip list`                        | List installed packages                             |
| `pip show <pkg>`                  | Package details                                     |
| `pip freeze`                      | List installed packages in `requirement.txt` format |
| `pip check`                       | Check for broken dependencies                       |

---

### File Formats

| File                   | Purpose                           |
| ---------------------- | --------------------------------- |
| `requirements.txt`     | List of packages to install       |
| `setup.py`             | Used in packaging Python projects |
| `pip.conf` / `pip.ini` | pip configuration file            |

---

### Options

| Option              | Description                    |
| ------------------- | ------------------------------ |
| `--upgrade`         | Upgrade to the latest version  |
| `--target <dir>`    | Install to specific directory  |
| `--user`            | Install to user site directory |
| `--no-cache-dir`    | Don’t use cache                |
| `--index-url <url>` | Use custom PyPI index          |
| `--proxy <url>`     | Use proxy                      |

---

### Upgrading `pip`

```bash
python -m pip install --upgrade pip
```

---

### Virtual Environment Usage

```bash
python -m venv env
source env/bin/activate   # Linux/macOS
env\Scripts\activate      # Windows
pip install <package>
```

---

### Trusted Indexes

```bash
pip install <pkg> --trusted-host <hostname> --index-url <url>
```

---

### Searching Packages (Deprecated)

```bash
pip search <keyword>  # Deprecated due to PyPI restrictions
```

Use:
Search on [https://pypi.org](https://pypi.org) directly.

---
