## Python Package Managers

### What is a Package Manager in Python?

A **package manager** is a tool that helps you **install, upgrade, remove, and manage external libraries (packages)** used in Python projects.

---

### Common Python Package Managers

| Package Manager | Description                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------ |
| **pip**         | Default Python package manager. Installs from [PyPI](https://pypi.org).                          |
| **conda**       | Part of Anaconda; manages both packages and environments. Supports Python & non-Python packages. |
| **poetry**      | Modern dependency manager with project management and environment isolation.                     |
| **pipenv**      | Combines `pip` & `virtualenv` for simplified dependency & environment management.                |
| **virtualenv**  | Tool to create isolated Python environments (often used with `pip`).                             |
| **venv**        | Built-in alternative to `virtualenv` for Python 3+.                                              |
| **buildout**    | Older, configuration-based tool mainly used in enterprise setups.                                |

---

### `pip` – Python’s Default Package Manager

#### Common Commands

```bash
pip install <package>            # Install a package
pip install <package>==1.2.3     # Install specific version
pip install -r requirements.txt  # Install from file
pip freeze > requirements.txt    # Save current packages to file
pip uninstall <package>          # Remove a package
pip list                         # List installed packages
pip show <package>               # Show details about a package
pip search <query>               # Search PyPI (deprecated)
```

#### Configuration

* Config files: `pip.conf` (Linux/macOS), `pip.ini` (Windows)
* Index override: `--index-url <url>` to use custom PyPI repository

---

### `conda` – Package & Environment Manager

#### Common Commands

```bash
conda install <package>            # Install a package
conda create -n myenv python=3.9   # Create new environment
conda activate myenv               # Activate environment
conda deactivate                   # Deactivate environment
conda list                         # List installed packages
conda remove <package>             # Uninstall a package
```

#### Supports binary dependencies (like C libraries), unlike `pip`.

---

### `poetry` – Modern Dependency Manager

#### Key Features

* Dependency resolution
* Virtualenv creation
* `pyproject.toml` based

```bash
poetry init                        # Start project config
poetry add <package>              # Add dependency
poetry install                    # Install all deps
poetry update                     # Update deps
poetry run <command>              # Run inside env
poetry shell                      # Open env shell
```

---

### `pipenv` – Combines pip + virtualenv

```bash
pipenv install <package>          # Install & add to Pipfile
pipenv shell                      # Enter virtualenv
pipenv run <command>              # Run command in env
pipenv graph                      # Show dependency tree
pipenv lock                       # Lock exact versions
```

* Uses `Pipfile` and `Pipfile.lock`

---

### `virtualenv` and `venv` – For Environment Isolation

| Tool         | Use Case                             |
| ------------ | ------------------------------------ |
| `virtualenv` | Works with Python 2/3; more features |
| `venv`       | Built-in from Python 3.3+            |

```bash
python -m venv env                 # Create env
source env/bin/activate           # Activate (Linux/macOS)
env\Scripts\activate              # Activate (Windows)
deactivate                        # Deactivate
```

---

### Dependency Specification Files

| File Name          | Used By         | Purpose                       |
| ------------------ | --------------- | ----------------------------- |
| `requirements.txt` | pip             | List of packages + versions   |
| `Pipfile`          | pipenv          | Track packages and settings   |
| `Pipfile.lock`     | pipenv          | Exact versions locked         |
| `pyproject.toml`   | poetry, PEP 518 | Unified build system metadata |

---

### Choosing the Right Tool

| Use Case                          | Suggested Tool   |
| --------------------------------- | ---------------- |
| Simple package installation       | pip + venv       |
| Data science with non-Py packages | conda            |
| Full project + dependency control | poetry or pipenv |
| Legacy systems                    | virtualenv       |

---
