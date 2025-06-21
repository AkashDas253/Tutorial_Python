## Markers

### Purpose

* Markers are used to add metadata to tests.
* They allow for grouping tests, skipping tests, expecting failures, and more.

### Built-in Markers

| Marker                     | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| `@pytest.mark.skip`        | Skips the test unconditionally                                 |
| `@pytest.mark.skipif`      | Skips the test conditionally based on an expression            |
| `@pytest.mark.xfail`       | Marks the test as expected to fail                             |
| `@pytest.mark.parametrize` | Parametrizes the test function with multiple sets of arguments |
| `@pytest.mark.usefixtures` | Specifies fixtures to be used in tests                         |

### Skipping Tests

* **Unconditionally skip a test**:

```python
@pytest.mark.skip(reason="Test not implemented yet")
def test_not_done():
    pass
```

* **Skip test conditionally**:

```python
@pytest.mark.skipif(condition, reason="Some reason")
def test_conditional_skip():
    pass
```

* **Skip based on pytest version**:

```python
@pytest.mark.skipif(pytest.__version__ < "6.0", reason="Requires pytest 6.0 or higher")
def test_version_dependent():
    pass
```

### Expected Failures

* **Marking a test as expected to fail** (xfail):

```python
@pytest.mark.xfail(reason="Known bug in version 1.0")
def test_known_bug():
    assert 1 == 2
```

* **Handling test outcome with `xfail`**:

  * If the test fails, it is marked as `xpassed` (unexpected pass).
  * If the test passes, it is marked as `xfailed` (expected fail).

### Custom Markers

* **Define a custom marker** in `pytest.ini`:

```ini
# pytest.ini
[pytest]
markers =
    slow: marks tests as slow
    network: marks tests requiring network access
```

* **Apply custom markers** to tests:

```python
@pytest.mark.slow
def test_long_running_task():
    assert long_running_task() == "done"
```

### Marking a Group of Tests

You can mark multiple tests with the same marker:

```python
@pytest.mark.slow
def test_task1():
    pass

@pytest.mark.slow
def test_task2():
    pass
```

### Running Tests Based on Markers

* Run tests with specific markers:

```bash
pytest -m "slow"
```

* Combine markers using logical expressions:

```bash
pytest -m "slow and not network"
```

### Markers in Command-Line Options

You can skip tests or run tests based on markers through command-line options:

```bash
pytest -m "not slow"
```

### Use of `@pytest.mark.usefixtures`

* Automatically apply fixtures to tests via markers:

```python
@pytest.mark.usefixtures("db_setup")
def test_database():
    pass
```

---