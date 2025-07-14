## Test Output in `unittest`

---

#### **Purpose**

The test output in `unittest` provides **feedback** on test execution, including which tests passed, failed, were skipped, or had errors. Output is produced by the **test runner**, typically `unittest.TextTestRunner`.

---

### **1. Default Output Format**

When using:

```bash
python -m unittest test_module.py
```

You typically get output like:

```
..
----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```

Each character represents a test:

* `.` – Test passed
* `F` – Test failed (assertion failed)
* `E` – Test error (exception occurred)
* `s` – Test skipped
* `x` – Expected failure
* `u` – Unexpected success

---

### **2. Verbosity Levels**

You can control output detail using the `verbosity` parameter in `TextTestRunner`.

#### Syntax:

```python
unittest.TextTestRunner(verbosity=1).run(suite)
```

| Verbosity | Level   | Output Description                  |
| --------- | ------- | ----------------------------------- |
| `0`       | Quiet   | Only summary, no test names         |
| `1`       | Default | Dots for test results               |
| `2`       | Verbose | Names of each test + result message |

---

### **3. Interpreting Output**

Example (`verbosity=2`):

```
test_addition (test_math.TestMath) ... ok
test_divide_by_zero (test_math.TestMath) ... ERROR
test_invalid_input (test_math.TestMath) ... FAIL
test_not_implemented (test_math.TestMath) ... skipped 'Not ready'

======================================================================
FAIL: test_invalid_input (test_math.TestMath)
Traceback (most recent call last):
  File "test_math.py", line 10, in test_invalid_input
    self.assertEqual(add('a', 1), 2)
AssertionError: 'a1' != 2

======================================================================
ERROR: test_divide_by_zero (test_math.TestMath)
Traceback (most recent call last):
  ...
ZeroDivisionError: division by zero

----------------------------------------------------------------------
Ran 4 tests in 0.003s

FAILED (failures=1, errors=1, skipped=1)
```

---

### **4. Programmatic Access to Output**

You can capture results using a `TestResult` object:

```python
suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print(result.wasSuccessful())          # True if all tests passed
print(result.errors)                   # List of (test, traceback)
print(result.failures)                 # List of (test, traceback)
print(result.skipped)                  # List of (test, reason)
print(result.expectedFailures)         # List of expected failures
print(result.unexpectedSuccesses)      # List of unexpected passes
```

---

### **5. Customizing Output**

You can create a custom test runner or result handler by subclassing:

```python
class MyTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"SUCCESS: {test}")

class MyTestRunner(unittest.TextTestRunner):
    resultclass = MyTestResult

MyTestRunner(verbosity=2).run(suite)
```

---

### **6. Output Redirection**

You can redirect output to a file:

```python
with open('results.txt', 'w') as f:
    runner = unittest.TextTestRunner(stream=f, verbosity=2)
    runner.run(suite)
```

---
