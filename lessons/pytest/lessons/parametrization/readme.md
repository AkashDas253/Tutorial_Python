## Parametrization

### Purpose

* Enables running the same test with different data inputs.
* Reduces code duplication by running a test function with multiple parameter sets.

### `pytest.mark.parametrize` Decorator

* Used to parametrize a test function or a fixture.
* Accepts a list of arguments and corresponding values.

### Function-level Parametrization

```python
import pytest

@pytest.mark.parametrize("a, b, expected", [(1, 2, 3), (2, 3, 5), (4, 5, 9)])
def test_add(a, b, expected):
    assert a + b == expected
```

* Each tuple in the list provides values for the parameters in the test function.
* The test is executed once for each parameter combination.

### Parametrization with Multiple Arguments

```python
@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("b", [4, 5])
def test_multiply(a, b):
    assert a * b in [4, 5, 8, 10, 12, 15]
```

* Multiple `parametrize` decorators can be stacked to create combinations of parameters.

### Parametrization with Indirect Fixtures

* Fixtures can also be parametrized by passing `indirect=True`:

```python
@pytest.fixture
def setup_data(request):
    return request.param * 2

@pytest.mark.parametrize("setup_data", [1, 2, 3], indirect=True)
def test_data(setup_data):
    assert setup_data in [2, 4, 6]
```

### Parametrization and Custom Markers

* Parametrize also works with custom markers for better organization:

```python
@pytest.mark.parametrize("number", [2, 4, 6], marks=pytest.mark.xfail)
def test_even(number):
    assert number % 2 == 0
```

### Skipping Tests Based on Parametrization

* Tests can be skipped for specific parameters using `@pytest.mark.skipif`:

```python
@pytest.mark.parametrize("num", [0, 1, 2, 3])
@pytest.mark.skipif(lambda num: num == 0, reason="Skip zero test")
def test_nonzero(num):
    assert num != 0
```

### Parametrization with Dynamic Values

* Parametrization can accept dynamic values from functions or fixtures.

```python
@pytest.mark.parametrize("value", get_values())
def test_dynamic(value):
    assert value in [1, 2, 3]
```

---