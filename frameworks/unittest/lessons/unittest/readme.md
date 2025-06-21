### Overview of `unittest` in Python

---

#### What is `unittest`?

* `unittest` is a built-in Python testing framework inspired by Java’s JUnit.
* It supports test automation, sharing setup and shutdown code, test discovery, and aggregation of tests into collections.

---

#### Core Concepts

* **TestCase**: Base class to write individual test cases.
* **TestSuite**: Combines multiple test cases.
* **TestLoader**: Loads test cases from a module or class.
* **TestRunner**: Executes tests and returns the result.
* **TestFixture**: Setup and teardown routines for test environments.

---

#### Writing a Test Case

```python
import unittest

class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Code to set up test environment
        pass

    def test_example(self):
        self.assertEqual(1 + 1, 2)

    def tearDown(self):
        # Code to clean up after test
        pass

if __name__ == '__main__':
    unittest.main()
```

---

#### Assertion Methods

| Method                      | Description                                |
| --------------------------- | ------------------------------------------ |
| `assertEqual(a, b)`         | Check `a == b`                             |
| `assertNotEqual(a, b)`      | Check `a != b`                             |
| `assertTrue(x)`             | Check that `x` is `True`                   |
| `assertFalse(x)`            | Check that `x` is `False`                  |
| `assertIs(a, b)`            | Check that `a is b`                        |
| `assertIsNot(a, b)`         | Check that `a is not b`                    |
| `assertIsNone(x)`           | Check `x is None`                          |
| `assertIsNotNone(x)`        | Check `x is not None`                      |
| `assertIn(a, b)`            | Check `a in b`                             |
| `assertNotIn(a, b)`         | Check `a not in b`                         |
| `assertIsInstance(a, b)`    | Check `isinstance(a, b)`                   |
| `assertNotIsInstance(a, b)` | Check `not isinstance(a, b)`               |
| `assertRaises(Exception)`   | Checks that a specific exception is raised |

---

#### Test Lifecycle Methods

| Method            | Purpose                                |
| ----------------- | -------------------------------------- |
| `setUp()`         | Run before each test method            |
| `tearDown()`      | Run after each test method             |
| `setUpClass()`    | Run once before all tests in the class |
| `tearDownClass()` | Run once after all tests in the class  |

---

#### Running Tests

* **Command Line**:

  ```bash
  python -m unittest test_module.py
  python -m unittest discover  # auto-discovers tests
  ```

* **In Code**:

  ```python
  unittest.main()
  ```

---

#### Organizing Tests

* **Structure**:

  ```
  /project
      /tests
          test_module1.py
          test_module2.py
      module1.py
      module2.py
  ```

* Use `unittest.TestLoader` to dynamically discover and load tests.

---

#### Mocking with `unittest.mock`

* `unittest.mock` allows you to replace parts of your system under test and make assertions about how they were used.

```python
from unittest.mock import patch

@patch('module.ClassName')
def test_mocking(mock_class):
    instance = mock_class.return_value
    instance.method.return_value = 'value'
```

---

#### Test Discovery

* Automatically finds all tests:

  ```bash
  python -m unittest discover -s tests -p "test_*.py"
  ```

---

#### Advanced Features

* **Subtests**:

  ```python
  for i in range(5):
      with self.subTest(i=i):
          self.assertEqual(i % 2, 0)
  ```

* **Skipping Tests**:

  ```python
  @unittest.skip("reason")
  @unittest.skipIf(condition, "reason")
  @unittest.expectedFailure
  ```

* **Custom Test Loaders and Runners** for advanced configurations

---

#### Use Cases

* Unit testing modules
* Regression testing
* Integration testing
* CI/CD testing pipelines

---
