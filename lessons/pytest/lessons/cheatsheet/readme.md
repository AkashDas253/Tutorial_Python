## Pytest Cheatsheet

### Test Discovery

* **Run all tests** in the current directory and subdirectories:

  ```bash
  pytest
  ```
* **Specify test files or directories**:

  ```bash
  pytest tests/test_file.py
  pytest tests/
  ```
* **Run specific test function** or method:

  ```bash
  pytest tests/test_file.py::test_function
  pytest tests/test_file.py::TestClass::test_method
  ```

### Command-Line Options

* **Verbose output** (show detailed test results):

  ```bash
  pytest -v
  ```
* **Disable output capturing** (for seeing print statements or logs):

  ```bash
  pytest -s
  ```
* **Stop after N failures**:

  ```bash
  pytest --maxfail=3
  ```
* **Run tests with specific markers**:

  ```bash
  pytest -m "smoke"
  pytest -m "not slow"
  ```
* **Change traceback style**:

  ```bash
  pytest --tb=short
  pytest --tb=long
  pytest --tb=no
  ```
* **Run tests in parallel (using `pytest-xdist`)**:

  ```bash
  pytest -n 4
  ```
* **Show slowest N tests**:

  ```bash
  pytest --durations=5
  ```
* **Generate test report in JUnit XML format**:

  ```bash
  pytest --junitxml=report.xml
  ```
* **Run with coverage reporting (requires `pytest-cov`)**:

  ```bash
  pytest --cov=my_module
  pytest --cov=my_module --cov-report=term
  ```
* **Enable debugging (PDB) on failure**:

  ```bash
  pytest --pdb
  ```
* **Disable warnings**:

  ```bash
  pytest -p no:warnings
  ```

### Markers

* **Define markers** using `@pytest.mark`:

  ```python
  @pytest.mark.smoke
  def test_smoke_case():
      assert True
  ```

* **Run tests with specific marker**:

  ```bash
  pytest -m "smoke"
  ```

### Fixtures

* **Basic fixture**:

  ```python
  @pytest.fixture
  def setup():
      return 42

  def test_example(setup):
      assert setup == 42
  ```

* **Async fixture**:

  ```python
  @pytest.fixture
  async def async_fixture():
      return 42

  @pytest.mark.asyncio
  async def test_async_example(async_fixture):
      assert async_fixture == 42
  ```

* **Fixture with setup/teardown**:

  ```python
  @pytest.fixture
  def resource():
      # Setup
      yield "resource"
      # Teardown
  ```

### Parametrization

* **Parametrize tests**:

  ```python
  @pytest.mark.parametrize("input, expected", [(1, 2), (2, 3)])
  def test_addition(input, expected):
      assert input + 1 == expected
  ```

* **Parametrize fixtures**:

  ```python
  @pytest.fixture(params=[1, 2, 3])
  def fixture_with_params(request):
      return request.param

  def test_with_param(fixture_with_params):
      assert fixture_with_params in [1, 2, 3]
  ```

### Assertions

* **Standard assertions**:

  ```python
  assert x == y
  assert x > 5
  assert x in [1, 2, 3]
  ```

* **Expected exception**:

  ```python
  with pytest.raises(ValueError):
      raise ValueError("Error!")
  ```

* **Custom assertion with message**:

  ```python
  assert x == 10, "x should be 10"
  ```

### Skipping and Expected Failures

* **Skip test**:

  ```python
  @pytest.mark.skip(reason="Not implemented yet")
  def test_to_be_skipped():
      assert False
  ```

* **Skip test conditionally**:

  ```python
  @pytest.mark.skipif(condition, reason="Condition met")
  def test_conditional_skip():
      assert False
  ```

* **Expected failures**:

  ```python
  @pytest.mark.xfail
  def test_expected_failure():
      assert False
  ```

### Plugins

* **Running tests with `pytest-xdist` for parallel execution**:

  ```bash
  pytest -n 4
  ```

* **Using `pytest-cov` for coverage reporting**:

  ```bash
  pytest --cov=my_module --cov-report=term
  ```

### Logging and Output

* **Display logs with different logging levels**:

  ```bash
  pytest --log-cli-level=DEBUG
  ```

### Test Selection

* **Run tests matching a substring in the name**:

  ```bash
  pytest -k "test_name"
  ```

### Test Dependency and Order

* **Control the order of tests** (using `pytest-ordering`):

  ```python
  @pytest.mark.first
  def test_first():
      assert True

  @pytest.mark.last
  def test_last():
      assert True
  ```

### Test Reporting

* **Generate test reports (JUnit XML format)**:

  ```bash
  pytest --junitxml=report.xml
  ```

* **Generate HTML reports**:

  ```bash
  pytest --html=report.html
  ```

---
