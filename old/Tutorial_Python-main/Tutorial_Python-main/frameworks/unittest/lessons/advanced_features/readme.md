## Advanced Features in `unittest`

---

#### **Purpose**

Beyond basic testing, `unittest` offers several advanced features to support complex, scalable, and maintainable test workflows—such as custom test loading, running, filtering, and dynamic testing behavior.

---

### **1. Custom Test Loaders**

You can create your own test discovery logic by subclassing `unittest.TestLoader`.

#### Syntax:

```python
class CustomLoader(unittest.TestLoader):
    def loadTestsFromTestCase(self, testCaseClass):
        # Custom logic here
        return super().loadTestsFromTestCase(testCaseClass)

loader = CustomLoader()
suite = loader.discover(start_dir='tests')
```

---

### **2. Custom Test Runners**

You can create a custom test runner by subclassing `unittest.TextTestRunner` and `unittest.TextTestResult`.

#### Syntax:

```python
class MyTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"✓ {test}")

class MyTestRunner(unittest.TextTestRunner):
    resultclass = MyTestResult

runner = MyTestRunner(verbosity=2)
runner.run(unittest.TestLoader().discover('tests'))
```

---

### **3. Subtests**

Allows you to write parameterized tests within a single method, with independent failure tracking.

#### Syntax:

```python
for i in range(5):
    with self.subTest(i=i):
        self.assertEqual(i % 2, 0)
```

---

### **4. Skipping and Expected Failures**

Decorators for dynamic or conditional test control:

* `@unittest.skip(reason)`
* `@unittest.skipIf(condition, reason)`
* `@unittest.skipUnless(condition, reason)`
* `@unittest.expectedFailure`

Also supported within methods:

```python
self.skipTest("reason")
```

---

### **5. Dynamic Test Case Generation**

You can dynamically create test methods or classes at runtime.

#### Syntax:

```python
def dynamic_test(self):
    self.assertTrue(True)

DynamicTest = type(
    'DynamicTest',
    (unittest.TestCase,),
    {'test_generated': dynamic_test}
)
```

---

### **6. Parallel Testing (External)**

Python 3.11+ includes `unittest` support for test parallelization:

```bash
python -m unittest discover -j 4  # Run with 4 parallel workers
```

For older versions, use third-party tools like:

* `pytest-xdist`
* `nose2.plugins.mp`

---

### **7. Test Filtering and Pattern Matching**

Run specific tests via CLI:

```bash
python -m unittest test_module.TestClass.test_method
```

Use `-k` for pattern matching in third-party runners (not `unittest` core).

---

### **8. CLI Output Redirection**

Run and write to file:

```bash
python -m unittest discover > results.txt
```

Or redirect via custom runner:

```python
with open("results.txt", "w") as f:
    unittest.TextTestRunner(stream=f).run(suite)
```

---

### **9. Testing with Context Managers**

`unittest` supports context-manager based assertions:

#### Exception assertion:

```python
with self.assertRaises(ValueError):
    int("x")
```

#### Warning assertion:

```python
with self.assertWarns(DeprecationWarning):
    deprecated_function()
```

---

### **10. Integration with External Tools**

* **Coverage**: `coverage run -m unittest`
* **CI/CD**: GitHub Actions, Jenkins, GitLab CI
* **Reporting**:

  * `unittest-xml-reporting`
  * `HtmlTestRunner`
  * `xmlrunner`

---
