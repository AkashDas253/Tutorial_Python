### All Concepts and Subconcepts in `unittest` 

---

#### Core Structure

* **Test Case (`TestCase`)**

  * Inheritance from `unittest.TestCase`
  * Defining test methods (`test_*`)
  * Using assertion methods
* **Test Suite (`TestSuite`)**

  * Combining multiple test cases
  * Manual aggregation
* **Test Loader (`TestLoader`)**

  * Load from module/class/name
  * `loadTestsFromTestCase`, `loadTestsFromModule`
* **Test Runner (`TextTestRunner`)**

  * Run test suite
  * Control verbosity and output
* **Test Result (`TestResult`)**

  * Captures test run outcomes
  * Tracks success, failure, errors, skipped

---

#### Test Lifecycle Hooks

* **Per-Test**

  * `setUp()`
  * `tearDown()`
* **Per-Class**

  * `setUpClass(cls)`
  * `tearDownClass(cls)`
* **Per-Module**

  * `setUpModule()`
  * `tearDownModule()`

---

#### Assertion Methods

* **Equality**

  * `assertEqual(a, b)`
  * `assertNotEqual(a, b)`
* **Truthiness**

  * `assertTrue(x)`
  * `assertFalse(x)`
* **Identity**

  * `assertIs(a, b)`
  * `assertIsNot(a, b)`
* **Nullity**

  * `assertIsNone(x)`
  * `assertIsNotNone(x)`
* **Containment**

  * `assertIn(a, b)`
  * `assertNotIn(a, b)`
* **Type Checking**

  * `assertIsInstance(a, b)`
  * `assertNotIsInstance(a, b)`
* **Exception Testing**

  * `assertRaises(Exception)`
  * `assertRaisesRegex(Exception, regex)`
* **Warnings**

  * `assertWarns`
  * `assertWarnsRegex`
* **Almost Equality**

  * `assertAlmostEqual`
  * `assertNotAlmostEqual`
* **Multiline String Equality**

  * `assertMultiLineEqual`

---

#### Test Discovery

* **Manual test discovery**

  * Using `TestLoader`
* **Automatic discovery**

  * Via `unittest discover`
  * Command line: `python -m unittest discover`
* **Pattern-based discovery**

  * `-p "test_*.py"`
* **Directory-based discovery**

  * `-s <directory>`

---

#### Skipping and Expected Failures

* **Skipping tests**

  * `@unittest.skip(reason)`
  * `@unittest.skipIf(condition, reason)`
  * `@unittest.skipUnless(condition, reason)`
* **Expected failures**

  * `@unittest.expectedFailure`

---

#### Mocking (via `unittest.mock`)

* **Patching**

  * `@patch(target)`
  * `patch.object()`
* **Mock Objects**

  * `Mock()`, `MagicMock()`
  * Setting return values, side effects
* **Assertions on mocks**

  * `assert_called_with()`
  * `assert_called_once()`
* **Specifying behavior**

  * `return_value`, `side_effect`
* **Nested mocking**

  * `patch.multiple`
* **Mock call history**

  * `mock.call_args`
  * `mock.call_count`

---

#### Subtests

* **Using subtests**

  * `with self.subTest(param=value):`
  * Useful for looping over similar assertions

---

#### Test Output

* **Verbosity levels**

  * `verbosity=0, 1, 2`
* **Result report**

  * Success, failure, error, skip
* **Custom output formatting**

  * Extend `TextTestResult`, `TextTestRunner`

---

#### Advanced Features

* **Custom Test Runners**

  * Inherit and extend `unittest.TextTestRunner`
* **Custom Test Loaders**

  * Custom logic to load and order tests
* **Test tagging (indirectly via naming or decorators)**

---

#### Integration

* **CI tools (Jenkins, GitHub Actions)**
* **Code coverage tools**

  * Compatible with `coverage.py`
* **Third-party test report generators**

  * `unittest-xml-reporting`
  * HTML reporting tools

---

#### Organizational Practices

* **Test structure**

  * Place tests in `/tests` or `/test` directory
  * Use `test_*.py` for filenames
* **Test naming**

  * Use `test_*` for functions
  * Use `Test*` for classes
* **Modularizing tests**

  * Grouping related tests per feature/module

---
