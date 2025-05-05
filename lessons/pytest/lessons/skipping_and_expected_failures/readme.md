## Skipping and Expected Failures in Pytest

### Purpose

Pytest provides mechanisms for skipping tests and marking tests as expected failures. These features help manage tests that are either not relevant in certain conditions or are known to fail, preventing them from causing unnecessary failures in the test suite.

### Skipping Tests

Sometimes, certain tests need to be skipped under specific conditions. Pytest provides the `@pytest.mark.skip` and `@pytest.mark.skipif` decorators to control when tests are skipped.

#### `@pytest.mark.skip`

The `@pytest.mark.skip` decorator is used to unconditionally skip a test. This is useful when a test is not applicable to the current environment or is temporarily disabled.

##### Example:

```python
import pytest

@pytest.mark.skip(reason="Test is not applicable")
def test_not_needed():
    assert False
```

In this example, the `test_not_needed` will be skipped and will not be executed. You can provide a `reason` for the skip to make it clear why the test is being skipped.

#### `@pytest.mark.skipif`

The `@pytest.mark.skipif` decorator allows you to skip a test conditionally based on a specified condition. The test will only be skipped if the condition evaluates to `True`.

##### Syntax:

```python
@pytest.mark.skipif(condition, reason="explanation")
```

##### Example:

```python
import pytest
import sys

@pytest.mark.skipif(sys.platform == "win32", reason="Test does not run on Windows")
def test_only_on_linux():
    assert True
```

In this example, the `test_only_on_linux` will be skipped if the platform is `win32` (Windows), with a reason explaining why it was skipped.

### Expected Failures

Sometimes, you may want to mark a test as an expected failure. This is useful when a test is known to fail due to a bug or limitation, but you don't want it to be considered as a failure in your test suite.

#### `@pytest.mark.xfail`

The `@pytest.mark.xfail` decorator marks a test as expected to fail. If the test fails, it will be reported as an "expected failure" rather than a regular failure. If the test passes, it will be reported as an "unexpected pass."

##### Syntax:

```python
@pytest.mark.xfail(reason="description")
```

##### Example:

```python
import pytest

@pytest.mark.xfail(reason="Bug in feature X")
def test_known_bug():
    assert 1 == 2
```

In this example:

* The test `test_known_bug` is expected to fail.
* If it fails, it will be reported as an expected failure and will not affect the overall test result.
* If it passes (which is unlikely), it will be reported as an "unexpected pass," indicating that the failure was fixed unexpectedly.

#### Handling Conditions with `xfail` and `skipif`

You can combine `@pytest.mark.xfail` with conditions using `@pytest.mark.skipif`. This allows you to mark a test as expected to fail only under certain conditions.

##### Example:

```python
import pytest
import sys

@pytest.mark.xfail(sys.platform == "win32", reason="Known issue on Windows")
def test_feature_on_windows():
    assert False  # This will fail if the platform is Windows
```

In this example, the test is marked as an expected failure on Windows platforms. If it fails on Windows, it will be reported as expected.

### Summary of Markers

* **`@pytest.mark.skip`**: Skips the test unconditionally.
* **`@pytest.mark.skipif(condition)`**: Skips the test conditionally based on the evaluation of the `condition`.
* **`@pytest.mark.xfail`**: Marks the test as expected to fail. If the test passes, it will be reported as an "unexpected pass."

### Useful Information During Skipping and Expected Failures

* Skipping and expected failures are logged and shown in the test results to provide context for why a test was skipped or expected to fail.
* The `-rs` option can be used with `pytest` to include extra information on skipped tests in the output.

### Example Output

When running tests with skipped or expected failures, the output might look like:

```
========================= test session starts ==========================
collected 4 items

test_example.py .xx.                                              [100%]

========================= short test summary info ==========================
SKIPPED: test_not_needed
XFAIL: test_known_bug
========================= 2 passed, 1 failed, 1 skipped in 0.12 seconds =========================
```

* `XFAIL` indicates an expected failure.
* `SKIPPED` indicates a skipped test.

### Conclusion

Skipping tests and marking tests as expected failures are powerful features in Pytest that help maintain clean and accurate test suites. These tools are particularly useful for handling tests that are temporarily not needed or tests that are known to fail due to existing issues. By using these decorators effectively, you can improve the readability and maintainability of your tests.

---