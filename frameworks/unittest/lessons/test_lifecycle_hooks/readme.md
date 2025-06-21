## Test Lifecycle Hooks in `unittest`

---

#### **Purpose**

Test lifecycle hooks manage **setup and teardown logic** to ensure that test environments are properly prepared and cleaned up before and after tests.

Lifecycle hooks can be applied:

* Per test method
* Per test class
* Per test module

---

### **1. Method-Level Hooks**

These are run **before and after each test method** in a class that inherits `unittest.TestCase`.

#### `setUp(self)`

* Called **before** each individual test method.
* Used to initialize test environment or variables.

#### `tearDown(self)`

* Called **after** each individual test method.
* Used to clean up resources or reset state.

**Syntax**:

```python
import unittest

class ExampleTest(unittest.TestCase):
    def setUp(self):
        # Runs before every test method
        self.resource = []

    def tearDown(self):
        # Runs after every test method
        self.resource.clear()
```

---

### **2. Class-Level Hooks**

These are run **once** for the **entire test class**, regardless of how many test methods it contains.

#### `@classmethod setUpClass(cls)`

* Called **once** before any test methods in the class run.
* Commonly used to set up expensive operations like database connections or shared state.

#### `@classmethod tearDownClass(cls)`

* Called **once** after all test methods in the class have run.
* Used to release shared resources or perform summary cleanups.

**Syntax**:

```python
class ExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Runs once before any test methods in this class
        cls.shared_resource = []

    @classmethod
    def tearDownClass(cls):
        # Runs once after all test methods in this class
        cls.shared_resource = None
```

---

### **3. Module-Level Hooks**

These hooks are not class-based and apply to the **entire module** (i.e., file-level).
They are useful for shared setup or teardown logic across multiple test classes.

#### `setUpModule()`

* Runs once **before** any test case in the module.

#### `tearDownModule()`

* Runs once **after** all tests in the module.

**Syntax**:

```python
def setUpModule():
    # Runs before any tests in this module
    print("Module-level setup")

def tearDownModule():
    # Runs after all tests in this module
    print("Module-level teardown")
```

---

### **4. Execution Order of Hooks**

1. `setUpModule()`
2. `setUpClass()`
3. `setUp()` → `test_*` → `tearDown()` (for each method)
4. `tearDownClass()`
5. `tearDownModule()`

---
