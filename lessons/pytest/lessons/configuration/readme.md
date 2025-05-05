## Pytest Configuration

### Purpose

* Pytest configuration allows users to customize and control how tests are executed, manage options like test discovery, plugins, and more.
* Configuration can be defined in several places, such as configuration files or command-line options.

### Configuration Files

Pytest supports configuration via the following files:

* `pytest.ini`: The most common configuration file for Pytest settings.
* `tox.ini`: Used for setting up testing environments with Tox, often in conjunction with Pytest.
* `setup.cfg`: Another common configuration file for managing testing setups.
* `pyproject.toml`: Can also be used for configuration in modern setups.

### `pytest.ini` File

The `pytest.ini` file is the main configuration file used to store Pytest settings, markers, and command-line options.

#### Example of `pytest.ini`

```ini
[pytest]
# General options
minversion = 6.0
addopts = --maxfail=5 --disable-warnings

# Markers
markers =
    slow: marks tests as slow
    integration: marks tests requiring external resources

# Custom plugins
python_files = test_*.py *_test.py

# Additional config
log_cli = true
log_cli_level = INFO
```

### Configuration Fields in `pytest.ini`

* **minversion**: Specifies the minimum version of Pytest required for the configuration file to be valid.

```ini
minversion = 6.0
```

* **addopts**: Specifies command-line options to be passed to Pytest each time it is run.

```ini
addopts = --maxfail=3 --disable-warnings
```

* **markers**: Defines custom markers, allowing you to categorize and organize your tests.

```ini
markers =
    slow: marks tests as slow
    integration: marks tests that require network or database access
```

* **python\_files**: Configures which files Pytest should treat as test modules.

```ini
python_files = test_*.py *_test.py
```

* **log\_cli and log\_cli\_level**: Configure logging directly within the test output.

```ini
log_cli = true
log_cli_level = INFO
```

### Configuration via Command-Line Options

Some configuration can be applied directly via command-line options, which override the settings in configuration files.

#### Common Command-Line Options

* **`-v` (Verbose)**: Increases output verbosity.

```bash
pytest -v
```

* **`--maxfail`**: Stop test execution after a specified number of failures.

```bash
pytest --maxfail=3
```

* **`--disable-warnings`**: Suppresses warnings in the test output.

```bash
pytest --disable-warnings
```

* **`-m`**: Run tests with a specific marker.

```bash
pytest -m "slow"
```

* **`-k`**: Run tests that match the given expression.

```bash
pytest -k "test_login"
```

* **`--tb`**: Set the traceback style for test failures. Options include `short`, `long`, and `line`.

```bash
pytest --tb=short
```

* **`--capture`**: Control output capturing behavior. Use `no` to disable capturing.

```bash
pytest --capture=no
```

### Configuration with `tox.ini`

* **Purpose**: `tox.ini` is used to configure multiple testing environments. This is commonly used for running tests across different Python versions.

#### Example of `tox.ini`

```ini
[tox]
envlist = py37, py38, py39

[testenv]
deps = pytest
commands = pytest tests/
```

* **envlist**: Specifies the environments to be tested.
* **testenv**: Defines the dependencies and commands for each environment.

### Configuration with `setup.cfg`

* **Purpose**: `setup.cfg` is often used for packaging Python projects and can also be used to configure test tools like Pytest.

#### Example of `setup.cfg`

```ini
[tool:pytest]
minversion = 6.0
addopts = --maxfail=5 --disable-warnings
```

### Configuration with `pyproject.toml`

* **Purpose**: `pyproject.toml` is part of PEP 518 and is increasingly used for configuring tools like Pytest.

#### Example of `pyproject.toml`

```toml
[tool.pytest]
minversion = "6.0"
addopts = "--maxfail=5 --disable-warnings"
```

### Configuration in `conftest.py`

* **Purpose**: `conftest.py` is used to store configuration hooks and fixture setup that should be applied globally across multiple test files.

```python
# conftest.py
import pytest

@pytest.fixture(scope="session")
def database_connection():
    return create_db_connection()

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
```

### `pytest` Environment Variables

You can also set environment variables for configuration, particularly for custom plugins and integration with other systems.

#### Example

```bash
export PYTHONPATH=src/
export DATABASE_URL="postgres://localhost/test"
pytest
```

---
