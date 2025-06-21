## Parametrized Fixtures

### Purpose

Parametrized fixtures in Pytest allow you to run the same test with different sets of data, which helps avoid redundant code and enhances test coverage. With parametrized fixtures, you can provide various values to a fixture, and the test will be executed once for each value.

### How Parametrized Fixtures Work

A parametrized fixture is created using the `@pytest.fixture` decorator along with the `params` argument. The `params` argument is a list, tuple, or any iterable that defines the different values to pass to the fixture.

Each time the test runs, Pytest will use one of the values from the iterable, inject it into the fixture, and execute the test with that value.

### Syntax

```python
@pytest.fixture(params=[value1, value2, value3])
def my_fixture(request):
    return request.param
```

* The `params` argument is an iterable containing values that will be passed to the fixture.
* The `request.param` gives the current value from the iterable.

### Example of Parametrized Fixture

```python
import pytest

# Parametrized fixture
@pytest.fixture(params=[1, 2, 3])
def number(request):
    return request.param

# Test using the fixture
def test_multiply(number):
    assert number * 2 == number + number
```

In this example:

* The `number` fixture is parametrized with values `[1, 2, 3]`.
* The test `test_multiply` will be executed 3 times, once for each value: `1`, `2`, and `3`.
* The `request.param` gives the current value passed to the fixture.

### Output

The test will run three times, once for each parameter:

```
======================== test session starts ========================
collected 3 items

test_example.py ...                                               [100%]

======================== 3 passed in 0.12 seconds ========================
```

### Parametrizing with Multiple Arguments

You can also pass multiple parameters to a fixture by passing a list of tuples, where each tuple contains multiple values.

```python
@pytest.fixture(params=[(1, 2), (3, 4), (5, 6)])
def pair(request):
    return request.param

def test_addition(pair):
    a, b = pair
    assert a + b == 3, "Test failed for pair {}".format(pair)
```

In this example, the test will run for each pair `(1, 2)`, `(3, 4)`, and `(5, 6)`.

### Parametrizing Test Functions with Fixtures

You can combine parametrized fixtures with test functions. Each time the test function runs, the fixture will be called with the next value from the `params` iterable.

```python
@pytest.mark.parametrize("a, b", [(1, 2), (3, 4)])
def test_add(a, b):
    assert a + b == 3 or 7
```

This approach is suitable when you want to parametrize the test inputs directly, without using a separate fixture.

### Using `indirect` Parametrization

If you want to parametrize the values in a way that the values are passed indirectly to the fixture (rather than directly to the test), you can use the `indirect` keyword argument.

#### Example

```python
@pytest.fixture
def multiply_by(request):
    return request.param * 2

@pytest.mark.parametrize("multiply_by", [2, 3, 4], indirect=True)
def test_multiplication(multiply_by):
    assert multiply_by in [4, 6, 8]
```

In this case, the value `2`, `3`, or `4` is passed to the fixture, and the fixture doubles it (multiplies by 2) before the test is run.

### Benefits of Parametrized Fixtures

* **Code Reusability**: Allows you to reuse the same test function with different input values, reducing code duplication.
* **Improved Test Coverage**: Testing with a variety of inputs helps to ensure that your code works in a wide range of scenarios.
* **Clearer Test Structure**: Makes the tests more readable and maintainable by avoiding complex loops and repetitive code.

### Conclusion

Parametrized fixtures in Pytest provide a flexible and efficient way to run tests with multiple sets of data. They are useful for improving test coverage, reusability, and reducing boilerplate code, while maintaining the clarity and readability of the test code.

---