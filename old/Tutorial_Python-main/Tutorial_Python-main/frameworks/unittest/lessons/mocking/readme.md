## Mocking in `unittest` (`unittest.mock`)

---

#### **Purpose**

`unittest.mock` is a library for:

* Replacing parts of your system under test with mock objects.
* Asserting how those parts are used (called, arguments, etc.).
* Simulating behaviors, side effects, or return values.

---

### **1. Mock and MagicMock**

#### `Mock()`

Creates a general-purpose mock object.

**Syntax**:

```python
from unittest.mock import Mock

mock_obj = Mock()
mock_obj.method.return_value = 42

result = mock_obj.method(1, 2)
mock_obj.method.assert_called_with(1, 2)
```

#### `MagicMock()`

Same as `Mock()` but supports Python "magic methods" like `__len__`, `__getitem__`, etc.

**Syntax**:

```python
from unittest.mock import MagicMock

mock_list = MagicMock()
len(mock_list)  # Works like __len__
```

---

### **2. Patching**

Temporarily replaces a target object or function with a mock.

#### `@patch(target)`

Replaces the target with a mock during the test.

**Syntax**:

```python
from unittest.mock import patch

@patch('module.ClassName')
def test_class(mock_class):
    instance = mock_class.return_value
    instance.method.return_value = 5
```

#### `patch()` as context manager

**Syntax**:

```python
with patch('module.function_name') as mock_func:
    mock_func.return_value = 'mocked'
```

#### `patch.object(obj, attribute)`

Patches an attribute of a specific object or class.

**Syntax**:

```python
patch.object(SomeClass, 'method_name', return_value=True)
```

#### `patch.dict`

Temporarily changes a dictionary (like `os.environ`).

**Syntax**:

```python
with patch.dict('os.environ', {'ENV': 'test'}):
    ...
```

---

### **3. Configuring Mocks**

#### Set return value

```python
mock.method.return_value = 123
```

#### Set side effect

* To raise exceptions or simulate sequential returns

```python
mock.method.side_effect = Exception("fail")
mock.method.side_effect = [1, 2, 3]  # Sequential returns
```

---

### **4. Assertion Methods on Mocks**

| Assertion                        | Description                     |
| -------------------------------- | ------------------------------- |
| `assert_called()`                | Called at least once            |
| `assert_called_once()`           | Called exactly once             |
| `assert_called_with(*args)`      | Called with specified args      |
| `assert_called_once_with(*args)` | Called once with specified args |
| `assert_any_call(*args)`         | Called with args at least once  |
| `assert_has_calls([calls])`      | Called with all calls in order  |
| `assert_not_called()`            | Was never called                |

**Example**:

```python
mock.method(10)
mock.method.assert_called_once_with(10)
```

---

### **5. Accessing Call Info**

#### Call count

```python
mock.method.call_count
```

#### Arguments of last call

```python
mock.method.call_args           # (args, kwargs)
mock.method.call_args_list      # List of all calls
```

#### Manual `call` comparison

```python
from unittest.mock import call
mock.method.assert_has_calls([call(1), call(2)])
```

---

### **6. Nested Patching**

Use multiple patches as decorators or in a single `patch.multiple()` call.

**Syntax**:

```python
@patch('module.func1')
@patch('module.func2')
def test_nested(mock_func2, mock_func1):
    ...
```

---

### **7. Autospeccing**

Ensures that the mock only allows attributes and methods that exist on the real object.

**Syntax**:

```python
mocked = patch('module.ClassName', autospec=True)
```

---
