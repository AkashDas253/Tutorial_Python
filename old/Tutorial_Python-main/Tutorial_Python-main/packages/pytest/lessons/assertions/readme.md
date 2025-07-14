## Pytest Assertions

### Native Python `assert` Statement

* Pytest enhances standard `assert` statements by rewriting them at runtime.
* On failure, it shows detailed introspection of expressions.

```python
def test_add():
    assert 2 + 3 == 5
    assert "abc".upper() == "ABC"
```

### Assertion Introspection

Pytest rewrites the assertion and shows values involved in failure:

```python
def test_fail():
    x = 2
    y = 3
    assert x == y
```

Output:

```
>       assert x == y
E       assert 2 == 3
```

### Comparisons Supported with Introspection

* Equality: `assert a == b`
* Inequality: `assert a != b`
* Greater/Less: `assert a > b`, `assert a < b`
* Membership: `assert x in y`, `assert x not in y`
* Identity: `assert a is b`, `assert a is not b`

### Asserting Exceptions

Use `pytest.raises()` as a context manager:

```python
import pytest

def divide(a, b):
    return a / b

def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
```

#### Accessing the Exception

```python
def test_message():
    with pytest.raises(ValueError, match="invalid"):
        raise ValueError("invalid input")
```

* `match="text"` ensures the error message contains the text (regex supported).
* You can also access the exception instance:

  ```python
  with pytest.raises(ValueError) as exc_info:
      raise ValueError("error")
  assert "error" in str(exc_info.value)
  ```

### Asserting Warnings

Use `pytest.warns()` to test warning messages:

```python
import warnings

def test_warns():
    with pytest.warns(UserWarning, match="deprecated"):
        warnings.warn("This is deprecated", UserWarning)
```

### Custom Assertion Message

```python
assert a == b, "Expected a to equal b"
```

### Assertions in Loops

Avoid using `assert` inside loops without descriptive messages, or use `subtests` (via `pytest-subtests`) for better clarity.

### Assertions and Logging

When assertions fail, captured logs are shown if logging is enabled with `--log-cli-level`.

### Disabling Assertion Rewriting

* Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to disable all plugins, including assertion rewriting.
* Not recommended unless you are debugging Pytest itself.

---