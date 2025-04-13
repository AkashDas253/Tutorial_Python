## All Concepts and Subconcepts of `unittest` in Python

### 📦 Test Structure
- **Test Module**
  - Python file containing test classes and methods
- **Test Class**
  - Inherits from `unittest.TestCase`
- **Test Method**
  - Method name starts with `test`
- **Test Function (Standalone)**
  - Used with `unittest.FunctionTestCase`

---

### 🧪 Core Components
- **TestCase**
  - Inheritance base for all test classes
  - Encapsulates individual test logic
- **TestSuite**
  - Groups multiple test cases or other test suites
- **TestLoader**
  - Loads tests from modules, classes, or functions
- **TestRunner**
  - Runs test suites
  - Outputs results to console or files

---

### 🛠 Lifecycle Methods
- **setUp(self)**
  - Runs before each test method
- **tearDown(self)**
  - Runs after each test method
- **setUpClass(cls)**
  - Runs once before all test methods (class-level)
- **tearDownClass(cls)**
  - Runs once after all test methods (class-level)

---

### ✅ Assertion Methods
- **Equality & Identity**
  - `assertEqual(a, b)`
  - `assertNotEqual(a, b)`
  - `assertIs(a, b)`
  - `assertIsNot(a, b)`
- **Truthiness**
  - `assertTrue(x)`
  - `assertFalse(x)`
  - `assertIsNone(x)`
  - `assertIsNotNone(x)`
- **Containment**
  - `assertIn(a, b)`
  - `assertNotIn(a, b)`
- **Type & Instance**
  - `assertIsInstance(obj, cls)`
  - `assertNotIsInstance(obj, cls)`
- **Exception Testing**
  - `assertRaises(exc, callable, *args)`
  - `assertRaisesRegex(exc, regex, callable, *args)`
- **Warnings Testing**
  - `assertWarns(warn, callable, *args)`
  - `assertWarnsRegex(warn, regex, callable, *args)`
- **Floating Point**
  - `assertAlmostEqual(a, b, places)`
  - `assertNotAlmostEqual(a, b, places)`
- **Comparison**
  - `assertGreater(a, b)`
  - `assertGreaterEqual(a, b)`
  - `assertLess(a, b)`
  - `assertLessEqual(a, b)`
- **Collection**
  - `assertCountEqual(a, b)`
  - `assertListEqual(a, b)`
  - `assertTupleEqual(a, b)`
  - `assertSetEqual(a, b)`
  - `assertDictEqual(a, b)`

---

### 🏃 Running Tests
- **Command Line Execution**
  - `python -m unittest`
  - `python -m unittest test_module`
  - `python -m unittest test_module.TestClass`
  - `python -m unittest test_module.TestClass.test_method`
- **Inside Script**
  - `unittest.main()`
  - `unittest.TextTestRunner().run(suite)`

---

### 🧰 Test Discovery
- **Automatic Test Discovery**
  - `python -m unittest discover`
- **Directory Options**
  - `-s` (start directory)
  - `-p` (pattern to match test files)
  - `-t` (top-level directory)

---

### 🧪 Fixtures and Cleanups
- **addCleanup(func, *args, **kwargs)**
  - Registers cleanup actions
- **doCleanups()**
  - Forces cleanup manually

---

### 🧩 Advanced Features
- **Skip Decorators**
  - `@unittest.skip(reason)`
  - `@unittest.skipIf(condition, reason)`
  - `@unittest.skipUnless(condition, reason)`
- **Expected Failures**
  - `@unittest.expectedFailure`
- **Subtests**
  - `with self.subTest(var=value):`
- **Mocking (via `unittest.mock`)**
  - `Mock`, `patch`, `MagicMock`, etc.

---

### 📁 Test Suite Composition
- **Manually Building Suites**
  - `suite = unittest.TestSuite()`
  - `suite.addTest(TestClass("test_method"))`
- **Loading from Classes/Modules**
  - `unittest.TestLoader().loadTestsFromTestCase(TestClass)`
  - `unittest.TestLoader().loadTestsFromModule(module)`

---

### 📤 Output and Result Handling
- **TextTestRunner**
  - Console output
- **TestResult**
  - Internal structure to collect pass/fail/error info
- **failfast**, **buffer**, **catchbreak**
  - Optional args to `unittest.main()` or `TextTestRunner()`

---

### 🔄 Integration
- **CI/CD Pipelines**
  - Works with GitHub Actions, Jenkins, etc.
- **IDE Integration**
  - Supported in PyCharm, VSCode, Eclipse-PyDev

---
