## Subtests in `unittest`

---

#### **Purpose**

Subtests allow **looping over multiple cases** inside a single test method without stopping on the first failure. Each iteration is treated as an independent subtest, helping identify **which cases failed and which passed**.

Introduced in Python 3.4 via `unittest.TestCase.subTest`.

---

### **Why Use Subtests?**

* Reduces duplication of test code
* Improves test readability
* Reports **all failed inputs** instead of stopping at the first one
* Keeps related assertions grouped logically

---

### **Syntax**

```python
import unittest

class SubtestExample(unittest.TestCase):
    def test_multiple_cases(self):
        for i in range(5):
            with self.subTest(i=i):
                self.assertEqual(i % 2, 0)
```

---

### **Output Behavior**

* If one subtest fails, others still continue.
* Each failed subtest appears **separately in the test report**.

---

### **Example Use Case: Parameterized Assertions**

```python
class TestMath(unittest.TestCase):
    def test_square(self):
        test_cases = [(2, 4), (3, 9), (4, 16), (5, 25)]

        for base, expected in test_cases:
            with self.subTest(base=base):
                self.assertEqual(base ** 2, expected)
```

---

### **Common Parameters in `subTest()`**

* You can pass any number of keyword arguments to `subTest()`
* These parameters are shown in the failure report for easy debugging

**Syntax**:

```python
with self.subTest(param1=value1, param2=value2):
    ...
```

---

### **Usage Scenarios**

* Testing multiple inputs/outputs
* Looping through edge cases
* Validating configurations or rules
* Comparing expected vs actual results in parameterized form

---

### **Caveats**

* Subtests don't support setup/teardown within `with self.subTest()`.
* All subtests share the same `setUp()`/`tearDown()` context.
* Failure in one subtest does **not abort** the rest of the test method.

---
