## Pytest Plugins

### Purpose

* Pytest plugins extend the functionality of Pytest.
* They provide additional features such as enhanced test reporting, test coverage, parallel test execution, and more.

### Installing Plugins

* **PyPI (Python Package Index)**: Most plugins are available through PyPI and can be installed using `pip`.

```bash
pip install pytest-<plugin_name>
```

* **Popular plugins**: Examples include `pytest-cov`, `pytest-xdist`, `pytest-mock`, etc.

### Enabling Plugins

* Once installed, Pytest automatically detects and loads the plugins.
* If a plugin is installed but not used, it won’t interfere with the test run.

### Listing Installed Plugins

* You can list all active plugins in your environment using:

```bash
pytest --plugins
```

### Commonly Used Plugins

#### `pytest-cov`

* **Purpose**: Measures code coverage during test runs.

```bash
pip install pytest-cov
```

* **Usage**: Run tests with coverage report:

```bash
pytest --cov=src
```

* **Additional options**:

```bash
pytest --cov=src --cov-report=html  # HTML report
```

#### `pytest-xdist`

* **Purpose**: Enables parallel test execution to speed up test runs.

```bash
pip install pytest-xdist
```

* **Usage**: Run tests in parallel across multiple CPUs.

```bash
pytest -n 4  # Run tests using 4 CPUs
```

#### `pytest-mock`

* **Purpose**: Provides easy access to the `unittest.mock` module for mocking.

```bash
pip install pytest-mock
```

* **Usage**: Use the `mocker` fixture to mock objects.

```python
def test_function(mocker):
    mock_func = mocker.patch('module.func')
    mock_func.return_value = 42
    assert module.func() == 42
```

#### `pytest-django`

* **Purpose**: Provides Django-specific testing tools, including fixtures for database setup and test client.

```bash
pip install pytest-django
```

* **Usage**: Add a Django setting and use `pytest.mark.django_db` for database interaction.

```python
@pytest.mark.django_db
def test_create_user():
    user = User.objects.create(username='testuser')
    assert user.username == 'testuser'
```

#### `pytest-flask`

* **Purpose**: Provides Flask-specific testing tools.

```bash
pip install pytest-flask
```

* **Usage**: Simplifies testing Flask apps, provides `client` fixture for HTTP requests.

```python
def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200
```

#### `pytest-html`

* **Purpose**: Generates an HTML report after test execution.

```bash
pip install pytest-html
```

* **Usage**: Generate a report in HTML format.

```bash
pytest --html=report.html
```

### Creating Custom Plugins

* **Plugin development**: You can create custom plugins by defining hooks and functions.
* **Defining hooks**: Use `pytest_plugins` in `conftest.py` to load your custom plugin.

```python
# conftest.py
def pytest_configure(config):
    print("Custom Pytest Plugin Loaded")
```

* **Use of hooks**: Hooks allow you to extend the Pytest functionality, e.g., modifying test results, configuring fixtures, or changing the behavior of the test run.

```python
# Example: Adding a custom hook to modify test results
def pytest_runtest_makereport(item, call):
    if call.excinfo is not None:
        print(f"Test {item.nodeid} failed!")
```

### Example of Custom Plugin Installation

1. **Create a plugin**: Define hooks and functions for custom functionality in a `plugin.py` file.
2. **Install your plugin**:

```bash
pip install .
```

3. **Use the plugin**: Once installed, Pytest will automatically detect it.

### Configuring Plugins via `pytest.ini`

* Some plugins provide configuration options in `pytest.ini` for fine-tuning their behavior.

```ini
# pytest.ini
[pytest]
addopts = --maxfail=3 --disable-warnings
```

### Deactivating Plugins

* **Disabling plugins**: You can disable a specific plugin for a test run using `-p no:<plugin_name>`:

```bash
pytest -p no:warnings
```

---