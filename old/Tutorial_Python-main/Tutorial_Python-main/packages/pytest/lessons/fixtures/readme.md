## Pytest Fixtures

### Purpose

* Reusable setup/teardown code for tests
* Supports parameterization, scope control, and dependency injection

### Basic Syntax

```python
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3]

def test_sum(sample_data):
    assert sum(sample_data) == 6
```

* Fixture name is passed as a test function argument.
* Pytest resolves and injects it automatically.

### Scope Options

| Scope      | Description                          |
| ---------- | ------------------------------------ |
| `function` | (default) Run once per test function |
| `class`    | Run once per test class              |
| `module`   | Run once per module                  |
| `package`  | Run once per package (Pytest 7.0+)   |
| `session`  | Run once per test session            |

```python
@pytest.fixture(scope="module")
def db_conn():
    return create_connection()
```

### Autouse Fixtures

* Automatically used without being requested
* Useful for setup that applies globally

```python
@pytest.fixture(autouse=True)
def setup_env():
    os.environ["MODE"] = "TEST"
```

### Yield Fixtures (Teardown Support)

```python
@pytest.fixture
def resource():
    setup()
    yield "resource"
    teardown()
```

### Using Fixtures in Other Fixtures

Fixtures can depend on each other:

```python
@pytest.fixture
def base():
    return 10

@pytest.fixture
def derived(base):
    return base + 5
```

### Accessing Fixtures Manually

Use `request.getfixturevalue()` to resolve fixtures dynamically:

```python
def test_dynamic(request):
    data = request.getfixturevalue("sample_data")
    assert data
```

### Conftest.py for Shared Fixtures

* Place common fixtures in `conftest.py`
* Automatically discovered by Pytest

```python
# conftest.py
@pytest.fixture
def user():
    return {"id": 1, "name": "Alice"}
```

### Parametrized Fixtures

```python
@pytest.fixture(params=[1, 2, 3])
def number(request):
    return request.param

def test_param(number):
    assert number in [1, 2, 3]
```

---