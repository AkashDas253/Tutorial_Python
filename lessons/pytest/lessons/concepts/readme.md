## Pytest Concepts and Subconcepts

### Basics

* Test Discovery

  * Naming conventions
  * File and function patterns
* Running Tests

  * `pytest` command
  * Command-line options

### Assertions

* Built-in `assert` statements
* Custom assertion introspection
* `pytest.raises`
* `pytest.warns`
* Exception info (`excinfo`)

### Fixtures

* Basic fixtures

  * `@pytest.fixture`
  * Scope: `function`, `class`, `module`, `package`, `session`
* Fixture parameters
* Autouse fixtures
* Conftest fixtures (`conftest.py`)
* Yield fixtures
* Fixture finalization (`request.addfinalizer`, `yield`)
* Dynamic fixtures with `request`

### Parametrization

* `@pytest.mark.parametrize`
* Multiple argument combinations
* Parametrizing fixtures
* Indirect parametrization

### Markers

* Built-in markers

  * `skip`, `skipif`, `xfail`
  * `usefixtures`
  * `filterwarnings`
* Custom markers (`@pytest.mark.custom`)
* Marker registration (`pytest.ini`)

### Test Organization

* Test classes
* Setup and teardown methods

  * `setup_function`, `teardown_function`
  * `setup_method`, `teardown_method`
  * `setup_class`, `teardown_class`
* Modular structure
* `conftest.py`

### Plugins

* Built-in plugins
* External plugins (`pytest-cov`, `pytest-mock`, `pytest-django`, etc.)
* Plugin hooks
* Writing custom plugins

### Configuration

* Configuration files

  * `pytest.ini`
  * `pyproject.toml`
  * `tox.ini`
* CLI options
* `addopts`, `markers`, `testpaths`

### Logging and Output

* Capturing output (`capsys`, `caplog`)
* Verbose mode (`-v`)
* Showing locals (`--showlocals`)
* Summary output control

### Test Reporting

* `--junitxml=report.xml`
* `--tb=short`, `--tb=long`, `--tb=no`
* Custom test report formats
* HTML reports (`pytest-html`)

### Test Coverage

* `pytest-cov`

  * Coverage report options
  * Coverage configuration

### Mocking

* `unittest.mock`
* `pytest-mock` plugin
* Patching (`mocker.patch`)

### Debugging

* `--pdb`
* `--trace`
* Dropping to interactive shell

### Parametrized Fixtures

* `pytest.fixture(params=[...])`
* Accessing parameter with `request.param`

### Skipping and Expected Failures

* `@pytest.mark.skip`
* `@pytest.mark.skipif(condition)`
* `@pytest.mark.xfail`
* `strict=True` option

### Test Selection

* Keyword expression (`-k`)
* Marker expression (`-m`)
* Directory/file/path filtering

### Test Dependency and Order

* `pytest-dependency`
* `pytest-order`

### Async Testing

* `pytest-asyncio`
* `pytest.mark.asyncio`

### CLI Utilities

* `--maxfail`
* `--disable-warnings`
* `--collect-only`
* `--fixtures`
* `--help`

---
