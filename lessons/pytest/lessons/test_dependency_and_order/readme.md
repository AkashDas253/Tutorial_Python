## Test Dependency and Order in Pytest

### Purpose

Test dependency and order control how tests are executed relative to one another. While Pytest encourages writing independent tests, there are cases where the execution order or dependencies between tests matter. Pytest provides several mechanisms for controlling test dependencies and execution order to meet specific needs.

### Test Dependencies

In general, Pytest does not support direct test dependencies (where one test depends on the outcome of another). This is because one of the core principles of unit testing is that tests should be independent and isolated. However, there are workarounds for cases where dependencies between tests are unavoidable.

#### 1. **Using `pytest-dependency` Plugin**

The `pytest-dependency` plugin allows you to define explicit dependencies between tests. With this plugin, you can mark a test as dependent on the successful execution of another test.

##### Installation:

```bash
pip install pytest-dependency
```

##### Usage:

You can define dependencies using the `@pytest.mark.dependency` marker.

#### Example:

```python
import pytest

@pytest.mark.dependency()
def test_a():
    assert True

@pytest.mark.dependency(depends=["test_a"])
def test_b():
    assert True

@pytest.mark.dependency(depends=["test_b"])
def test_c():
    assert True
```

* In this example, `test_b` depends on `test_a` and will only run if `test_a` passes. Similarly, `test_c` depends on `test_b`.

If `test_a` fails, `test_b` and `test_c` will be skipped.

#### 2. **Using `pytest.mark.skipif` for Conditional Dependencies**

While not strictly dependencies, you can conditionally skip tests based on certain conditions. This can be useful if you need to ensure certain tests only run when others have passed or based on certain configurations.

#### Example:

```python
import pytest
import sys

@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_feature():
    assert True
```

### Test Order

By default, Pytest executes tests in the order they are discovered. However, there are scenarios where you might need to control the order of execution.

#### 1. **Using `pytest-ordering` Plugin**

The `pytest-ordering` plugin provides a way to specify the order in which tests should run.

##### Installation:

```bash
pip install pytest-ordering
```

##### Usage:

You can use the `@pytest.mark.run` decorator to specify the order of the tests. The test order is defined by an integer value.

#### Example:

```python
import pytest

@pytest.mark.run(order=2)
def test_second():
    assert True

@pytest.mark.run(order=1)
def test_first():
    assert True

@pytest.mark.run(order=3)
def test_third():
    assert True
```

* In this example, the tests will run in the order of their `order` values: `test_first`, `test_second`, and `test_third`.

#### 2. **Using `pytest.mark.parametrize` to Run in Specific Order**

In some cases, you may want to control the order of tests within a parametrized test function. Although `pytest.mark.parametrize` does not directly provide an order option, you can manually manage the order of parameters using sorting or specific list definitions.

#### Example:

```python
import pytest

@pytest.mark.parametrize("input,expected", [(2, 4), (1, 1), (3, 9)], ids=["test_2", "test_1", "test_3"])
def test_square(input, expected):
    assert input ** 2 == expected
```

* Here, you can control the order of the tests by sorting or manually specifying the parameter sets.

#### 3. **Using `pytest.mark.run` for Conditional Ordering**

You can also combine `@pytest.mark.run` with conditions or fixtures to control test execution dynamically.

#### Example:

```python
import pytest

@pytest.mark.run(order=1)
def test_one():
    assert True

@pytest.mark.run(order=2)
def test_two():
    assert True

@pytest.mark.run(order=3)
def test_three():
    assert True
```

Here, the order of test execution is controlled explicitly by the `order` values provided in the decorators.

### Skipping Tests Based on Dependencies

Sometimes, tests that are dependent on other tests may need to be skipped if the dependency fails or is skipped. Pytest provides the `@pytest.mark.skipif` decorator to handle such cases.

#### Example:

```python
import pytest

@pytest.mark.skipif(not condition, reason="Condition not met")
def test_dependency():
    assert True
```

This can be used to conditionally skip a test based on a dependency.

### Conclusion

Test dependency and order in Pytest allow for more complex testing scenarios where the execution flow between tests matters. While it’s generally encouraged to keep tests independent, tools like the `pytest-dependency` plugin and `pytest-ordering` allow for controlled execution of tests based on their dependencies or specified order. These features help in cases where test isolation is difficult or when certain tests must run in a sequence due to application logic.

---