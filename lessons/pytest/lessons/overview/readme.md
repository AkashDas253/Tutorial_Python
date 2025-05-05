## Comprehensive Overview of PyTest

### ⚙️ Philosophy and Design Principles

* **Simplicity First**: Uses native `assert` and Pythonic constructs.
* **Extensibility**: Via fixtures, plugins, and hooks.
* **Auto-Discovery**: Finds tests automatically using naming conventions.
* **Readability**: Clean output, clear error reports, and minimal boilerplate.

---

### 🧱 Architecture

| Component          | Role                                                             |
| ------------------ | ---------------------------------------------------------------- |
| Test Discovery     | Finds test files, classes, and functions matching patterns       |
| Test Collector     | Builds a collection tree of test cases                           |
| Test Runner        | Executes tests, manages setup/teardown, fixtures, and assertions |
| Assertion Rewriter | Enhances Python `assert` to show expression introspection        |
| Hooks & Plugins    | Allows extension and customization of Pytest’s behavior          |
| Reporting System   | Captures results, failures, and outputs in various formats       |

---

### 🧪 Test Discovery

* Files must start with `test_` or end with `_test.py`.
* Functions must start with `test_`.
* Classes must be named `Test*` and not include `__init__()`.

---

### 🔁 Test Execution

```bash
pytest                             # Run all tests
pytest test_file.py                # Run specific file
pytest -k "expression"             # Filter by test name
pytest -m "markername"             # Run tests with specific marker
```

---

### ✅ Assertions

* Native Python `assert` is enhanced to show intermediate values.
* Supports:

  * `assert expr`
  * `pytest.raises(Exception)`
  * `pytest.warns(Warning)`
* Custom introspection makes debugging easier.

---

### 🔧 Fixtures

* Provide setup/teardown logic
* Defined using `@pytest.fixture`
* Can be reused across tests
* Fixture scopes:

  * `function`, `class`, `module`, `package`, `session`
* Support finalization using `yield` or `request.addfinalizer`

```python
@pytest.fixture
def db():
    connect = setup_db()
    yield connect
    connect.close()
```

---

### 📌 Parametrization

* Enables data-driven testing
* Function-level:

  ```python
  @pytest.mark.parametrize("a,b,result", [(1,2,3), (3,4,7)])
  def test_add(a, b, result):
      assert a + b == result
  ```
* Fixture-level:

  ```python
  @pytest.fixture(params=[0, 1])
  def val(request):
      return request.param
  ```

---

### 🏷️ Markers

* Add metadata to tests
* Common markers:

  * `@pytest.mark.skip`
  * `@pytest.mark.skipif(cond)`
  * `@pytest.mark.xfail`
  * `@pytest.mark.parametrize`
* Must be registered in `pytest.ini` to avoid warnings.

---

### 📂 Test Organization

* Group using:

  * Test classes
  * Folders with `__init__.py` or `conftest.py`
* `conftest.py` is used to share fixtures/configs locally.

---

### 🛠️ Configuration

* Config files:

  * `pytest.ini`
  * `pyproject.toml`
  * `tox.ini`
* Key options:

  * `addopts`, `testpaths`, `markers`, `log_cli`, `log_level`

---

### 🔌 Plugins and Extensibility

* Plugin system uses **setuptools entry points**
* Built-in and external plugins available

  * `pytest-cov`, `pytest-mock`, `pytest-django`, `pytest-xdist`
* Write custom plugins using `pytest_addoption`, `pytest_runtest_setup`, etc.

---

### 📃 Reporting & Logging

* Output options:

  * `-v`, `--tb=short`, `--tb=long`, `--maxfail`, `--showlocals`
* Log capture:

  * `caplog`, `capsys` fixtures
* Exporting:

  * `--junitxml=report.xml`
  * `pytest-html` for HTML reports

---

### 🔁 Test Reordering & Dependencies

* Use `pytest-order` to set test order.
* Use `pytest-dependency` to define dependencies between tests.

---

### 🔍 Debugging Support

* `--pdb`: Drops into debugger on failure
* `--trace`: Stops before each test
* Works with `pdb`, `ipdb`, or `pudb`

---

### 🔄 Coverage

* Use `pytest-cov`

  * CLI: `pytest --cov=yourmodule`
  * Options: `--cov-report=term`, `--cov-fail-under=90`

---

### 🤖 Mocking

* Built-in: `unittest.mock`
* With plugin: `pytest-mock`

  ```python
  def test_func(mocker):
      mock = mocker.patch('module.Class')
  ```

---

### 🔗 Async Testing

* Use `pytest-asyncio` for testing coroutines
* Decorate tests with `@pytest.mark.asyncio`

---

### 🧩 Hooks

* Pytest allows plugin-style customization using **hooks**
* Examples:

  * `pytest_runtest_setup`
  * `pytest_collection_modifyitems`
  * `pytest_addoption`

---

### 📁 Common CLI Options

| Option               | Description                 |      |                 |
| -------------------- | --------------------------- | ---- | --------------- |
| `-v`                 | Verbose output              |      |                 |
| `-q`                 | Quiet mode                  |      |                 |
| `-x`                 | Stop after first failure    |      |                 |
| `--maxfail=N`        | Stop after N failures       |      |                 |
| \`--tb=short         | long                        | no\` | Traceback style |
| `--disable-warnings` | Suppress warnings           |      |                 |
| `--junitxml=path`    | Export test result as XML   |      |                 |
| `--collect-only`     | Show what tests will run    |      |                 |
| `--fixtures`         | Show all available fixtures |      |                 |

---

### 🧪 Sample Test Suite Structure

```
project/
├── tests/
│   ├── test_models.py
│   ├── test_views.py
│   └── conftest.py
├── src/
│   └── mycode.py
├── pytest.ini
```

---
