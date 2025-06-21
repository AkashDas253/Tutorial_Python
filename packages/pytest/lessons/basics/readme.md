## Pytest Basics

### Test Discovery

* **File naming**: Pytest discovers files matching the pattern `test_*.py` or `*_test.py`.
* **Function naming**: Test functions must be named with the `test_` prefix.
* **Class naming**:

  * Must start with `Test`
  * Should not have `__init__()` method
  * Only instance methods prefixed with `test_` are collected.

### Running Tests

* Run all tests in current directory:

  ```
  pytest
  ```
* Run tests in a specific file:

  ```
  pytest test_file.py
  ```
* Run a specific test function:

  ```
  pytest test_file.py::test_func_name
  ```

### Command-Line Options

| Option               | Description                                |      |                         |
| -------------------- | ------------------------------------------ | ---- | ----------------------- |
| `-v`                 | Verbose output                             |      |                         |
| `-q`                 | Quiet output                               |      |                         |
| `-x`                 | Stop after first failure                   |      |                         |
| `--maxfail=num`      | Stop after N failures                      |      |                         |
| \`--tb=short         | long                                       | no\` | Control traceback style |
| `-k "expression"`    | Run tests with name matching expression    |      |                         |
| `-m "marker"`        | Run tests with given marker                |      |                         |
| `--disable-warnings` | Ignore warnings                            |      |                         |
| `--collect-only`     | Show discovered tests without running them |      |                         |

### Return Codes

| Code | Meaning                    |
| ---- | -------------------------- |
| 0    | All tests passed           |
| 1    | Tests failed               |
| 2    | Interrupted                |
| 3    | Internal error             |
| 4    | Usage error (bad CLI args) |
| 5    | No tests collected         |

### Test Output and Summary

* Shows summary of passed, failed, skipped, xfailed, and xpassed tests
* Shows line number and file where failure occurred
* Optionally shows local variable values during failure using `--showlocals`

### Test Path Control

* Use the `--testpaths` option in `pytest.ini` to specify directories

  ```ini
  [pytest]
  testpaths = tests integration
  ```

---