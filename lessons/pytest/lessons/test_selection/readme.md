## Test Selection in Pytest

### Purpose

Test selection in Pytest refers to the process of choosing specific tests to run from a larger test suite. This is particularly useful when you want to run only a subset of tests based on various criteria like test names, tags, or test markers. It helps improve efficiency by allowing you to focus on particular tests during development or debugging.

### Ways to Select Tests

Pytest provides several methods to select and run specific tests from the test suite:

### 1. **Running Tests by Name**

You can run a specific test by providing the test function name after the file name.

#### Example:

```bash
pytest test_example.py::test_function_name
```

* This will run only `test_function_name` from `test_example.py`.

### 2. **Running Tests by Keyword Expression**

You can run tests that match a keyword expression in their name or path. This can be done using the `-k` option followed by the expression.

#### Syntax:

```bash
pytest -k "expression"
```

#### Example:

```bash
pytest -k "test_addition"
```

* This will run all tests whose name includes `test_addition`.

#### Example with multiple conditions:

```bash
pytest -k "addition or subtraction"
```

* This will run all tests whose name includes either `addition` or `subtraction`.

### 3. **Running Tests by Markers**

Pytest allows you to assign markers to tests, and you can then select tests based on those markers. Markers can be used to categorize or group tests based on certain criteria (e.g., `slow`, `smoke`, `integration`, etc.).

#### Syntax:

```bash
pytest -m marker_name
```

#### Example:

```bash
@pytest.mark.slow
def test_long_running_process():
    pass

# To run only tests marked with 'slow'
pytest -m slow
```

* The `-m` option allows you to run only those tests that are marked with the specified marker.

### 4. **Running Tests from a Specific File or Directory**

You can specify a file or directory to run all the tests contained within it.

#### Example:

```bash
pytest test_example.py
```

* This runs all tests in the `test_example.py` file.

#### Example for running tests from a directory:

```bash
pytest tests/
```

* This runs all tests inside the `tests/` directory.

### 5. **Running Tests Based on Test Class**

If you organize your tests in classes, you can run all tests from a specific class.

#### Syntax:

```bash
pytest test_example.py::TestClass
```

#### Example:

```bash
pytest test_example.py::TestMath
```

* This will run all tests inside the `TestMath` class.

### 6. **Combining Test Selection Criteria**

You can combine multiple test selection methods. For example, you can select tests by class, then filter them by name or marker.

#### Example:

```bash
pytest test_example.py::TestMath -k "addition" -m "smoke"
```

* This will run only tests in the `TestMath` class that include "addition" in their name and are marked with `smoke`.

### 7. **Running Tests Based on Line Numbers**

You can run a test based on the line number within a test file.

#### Syntax:

```bash
pytest test_example.py::test_function_name::<line_number>
```

#### Example:

```bash
pytest test_example.py::test_addition::5
```

* This runs the test defined on line 5 in `test_example.py`.

### 8. **Including/Excluding Specific Tests Using `--deselect`**

Pytest allows you to deselect tests (i.e., exclude tests from the selection). This is helpful if you want to exclude certain tests but still select others.

#### Syntax:

```bash
pytest --deselect <test_name_or_marker>
```

#### Example:

```bash
pytest --deselect test_example.py::test_function_name
```

* This will run all tests except `test_function_name` in the `test_example.py`.

### 9. **Running Tests Based on Test Failures**

You can re-run only the tests that failed in the previous test session using the `--last-failed` option.

#### Example:

```bash
pytest --last-failed
```

* This will run only the tests that failed in the last run.

### 10. **Limit the Number of Tests with `--maxfail`**

You can specify a maximum number of test failures. After this number is reached, Pytest will stop running further tests.

#### Syntax:

```bash
pytest --maxfail=3
```

* This will stop the test run after 3 failures.

### Conclusion

Test selection in Pytest helps to optimize your testing process by allowing you to run only the relevant tests based on names, markers, file paths, or other criteria. You can also combine different test selection strategies to further refine your test runs. Whether you're focusing on specific areas of the code or excluding certain tests, Pytest provides flexible ways to control which tests to execute, making the testing process faster and more efficient.

---