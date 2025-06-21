## Skipping and Expected Failures in `unittest`

---

#### **Purpose**

`unittest` provides decorators to:

* **Skip** tests that are not relevant or not ready.
* **Conditionally skip** tests based on runtime checks.
* **Mark tests as expected to fail**, which avoids reporting them as errors.

---

### **1. Skipping Tests**

---

#### **@unittest.skip(reason)**

Skips the test unconditionally.

**Syntax**:

```python
@unittest.skip("Feature not implemented yet")
def test_feature(self):
    ...
```

---

#### **@unittest.skipIf(condition, reason)**

Skips the test **if the condition is true**.

**Syntax**:

```python
@unittest.skipIf(sys.platform == 'win32', "Not supported on Windows")
def test_linux_only(self):
    ...
```

---

#### **@unittest.skipUnless(condition, reason)**

Skips the test **unless the condition is true**.

**Syntax**:

```python
@unittest.skipUnless(has_gpu_support(), "GPU required")
def test_gpu_computation(self):
    ...
```

---

### **2. Expected Failures**

---

#### **@unittest.expectedFailure**

Marks the test as **expected to fail**.

* If it fails → marked as "expected failure" (not a failure).
* If it passes → marked as "unexpected success".

**Syntax**:

```python
@unittest.expectedFailure
def test_unstable_feature(self):
    self.assertEqual(compute(), 42)
```

---

### **3. Skipping Within Test Code**

You can also skip dynamically during test execution using `self.skipTest(reason)`.

**Syntax**:

```python
def test_conditional(self):
    if not is_ready():
        self.skipTest("Precondition not met")
    self.assertTrue(True)
```

---

### **4. Test Results Interpretation**

| Test Outcome           | Description                            |
| ---------------------- | -------------------------------------- |
| **ok**                 | Test passed                            |
| **skipped**            | Test skipped                           |
| **expected failure**   | Test failed but was marked as expected |
| **unexpected success** | Test passed but was expected to fail   |

---
