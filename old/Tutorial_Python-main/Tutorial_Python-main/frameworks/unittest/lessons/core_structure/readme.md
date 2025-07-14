## Core Structure in `unittest`

---

#### **Test Case (`TestCase`)**

A **test case** is a single unit of testing. It checks for a specific response to a set of inputs.

**Key Points**:

* Inherit from `unittest.TestCase`
* Define methods starting with `test_`
* Use `assert` methods for validation

**Syntax**:

```python
import unittest

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(1 + 1, 2)  # Assertion

if __name__ == '__main__':
    unittest.main()
```

---

#### **Test Suite (`TestSuite`)**

A **test suite** is a collection of test cases or test suites. It allows grouping of tests to execute them together.

**Syntax**:

```python
import unittest

# Create a suite manually
suite = unittest.TestSuite()
suite.addTest(MyTestCase('test_something'))
suite.addTest(MyTestCase('test_another'))

# Run the suite
runner = unittest.TextTestRunner()
runner.run(suite)
```

---

#### **Test Loader (`TestLoader`)**

A **test loader** is responsible for loading test cases from test classes or modules.

**Syntax**:

```python
loader = unittest.TestLoader()
suite = loader.loadTestsFromTestCase(MyTestCase)
```

Load from module:

```python
import test_module
suite = loader.loadTestsFromModule(test_module)
```

---

#### **Test Runner (`TextTestRunner`)**

The **test runner** is responsible for running the test suite and producing results.

**Syntax**:

```python
runner = unittest.TextTestRunner(verbosity=2)  # verbosity: 0, 1, or 2
runner.run(suite)
```

---

#### **Test Result (`TestResult`)**

**TestResult** stores information about the outcomes of tests:

* Number of tests run
* Failures
* Errors
* Skipped tests

**Usage with Runner** (indirect use):

```python
result = runner.run(suite)
print(result.wasSuccessful())  # Returns True if all tests passed
```

For custom handling:

```python
result = unittest.TestResult()
suite.run(result)
```

---

#### **Test Fixtures**

Fixtures manage setup and cleanup for test cases.

**Methods**:

* `setUp(self)`: Runs before every test method
* `tearDown(self)`: Runs after every test method
* `setUpClass(cls)`: Runs once before all tests (class-level)
* `tearDownClass(cls)`: Runs once after all tests (class-level)

**Syntax**:

```python
class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup shared across all test methods
        pass

    def setUp(self):
        # Setup before each test method
        pass

    def test_example(self):
        pass

    def tearDown(self):
        # Cleanup after each test method
        pass

    @classmethod
    def tearDownClass(cls):
        # Cleanup after all test methods
        pass
```

---
