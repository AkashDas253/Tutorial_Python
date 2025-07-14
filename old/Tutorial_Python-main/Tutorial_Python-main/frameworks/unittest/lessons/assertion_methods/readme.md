## Assertion Methods in `unittest`

---

#### **Purpose**

Assertion methods in `unittest.TestCase` are used to **verify expected outcomes**. If an assertion fails, the test fails.

---

### **1. Equality Assertions**

| Method                 | Description    | Syntax Example                         |
| ---------------------- | -------------- | -------------------------------------- |
| `assertEqual(a, b)`    | Check `a == b` | `self.assertEqual(x, 5)`               |
| `assertNotEqual(a, b)` | Check `a != b` | `self.assertNotEqual(status, 'error')` |

---

### **2. Truth Value Assertions**

| Method           | Description               | Syntax Example              |
| ---------------- | ------------------------- | --------------------------- |
| `assertTrue(x)`  | Check that `x` is `True`  | `self.assertTrue(is_valid)` |
| `assertFalse(x)` | Check that `x` is `False` | `self.assertFalse(flag)`    |

---

### **3. Identity Assertions**

| Method              | Description        | Syntax Example                 |
| ------------------- | ------------------ | ------------------------------ |
| `assertIs(a, b)`    | Check `a is b`     | `self.assertIs(obj1, obj2)`    |
| `assertIsNot(a, b)` | Check `a is not b` | `self.assertIsNot(obj1, obj2)` |

---

### **4. None Assertions**

| Method               | Description           | Syntax Example               |
| -------------------- | --------------------- | ---------------------------- |
| `assertIsNone(x)`    | Check `x is None`     | `self.assertIsNone(result)`  |
| `assertIsNotNone(x)` | Check `x is not None` | `self.assertIsNotNone(data)` |

---

### **5. Membership Assertions**

| Method              | Description        | Syntax Example                |
| ------------------- | ------------------ | ----------------------------- |
| `assertIn(a, b)`    | Check `a in b`     | `self.assertIn(5, [1, 2, 5])` |
| `assertNotIn(a, b)` | Check `a not in b` | `self.assertNotIn('x', text)` |

---

### **6. Type Assertions**

| Method                      | Description                  | Syntax Example                       |
| --------------------------- | ---------------------------- | ------------------------------------ |
| `assertIsInstance(a, b)`    | Check `isinstance(a, b)`     | `self.assertIsInstance(val, int)`    |
| `assertNotIsInstance(a, b)` | Check `not isinstance(a, b)` | `self.assertNotIsInstance(val, str)` |

---

### **7. Exception Assertions**

| Method                           | Description                                  | Syntax Example                                                              |
| -------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------- |
| `assertRaises(exc, func, *args)` | Check if `func` raises `exc`                 | `self.assertRaises(ValueError, int, 'x')`                                   |
| `assertRaisesRegex(exc, regex)`  | Check if `exc` is raised and matches `regex` | `with self.assertRaisesRegex(ValueError, 'invalid literal'):`<br>`int('x')` |

---

### **8. Warning Assertions**

| Method                          | Description                         | Syntax Example                                                            |
| ------------------------------- | ----------------------------------- | ------------------------------------------------------------------------- |
| `assertWarns(warn)`             | Check if warning is issued          | `with self.assertWarns(DeprecationWarning):`<br>`deprecated_func()`       |
| `assertWarnsRegex(warn, regex)` | Check warning message matches regex | `with self.assertWarnsRegex(UserWarning, 'deprecated'):`<br>`warn_func()` |

---

### **9. Numeric Assertions**

| Method                       | Description                                          | Syntax Example                               |
| ---------------------------- | ---------------------------------------------------- | -------------------------------------------- |
| `assertAlmostEqual(a, b)`    | Check if `a ≈ b` (up to 7 decimal places by default) | `self.assertAlmostEqual(0.1 + 0.2, 0.3)`     |
| `assertNotAlmostEqual(a, b)` | Opposite of above                                    | `self.assertNotAlmostEqual(0.1 + 0.2, 0.31)` |

**Keyword args**:

```python
# Tolerance control
self.assertAlmostEqual(a, b, places=3)
self.assertAlmostEqual(a, b, delta=0.01)
```

---

### **10. Sequence & Container Assertions**

| Method                      | Description                  | Syntax Example                             |
| --------------------------- | ---------------------------- | ------------------------------------------ |
| `assertSequenceEqual(a, b)` | Check if sequences are equal | `self.assertSequenceEqual([1, 2], [1, 2])` |
| `assertListEqual(a, b)`     | Check if lists are equal     | `self.assertListEqual(list1, list2)`       |
| `assertTupleEqual(a, b)`    | Check if tuples are equal    | `self.assertTupleEqual(t1, t2)`            |
| `assertSetEqual(a, b)`      | Check if sets are equal      | `self.assertSetEqual(set1, set2)`          |
| `assertDictEqual(a, b)`     | Check if dicts are equal     | `self.assertDictEqual(d1, d2)`             |

---

### **11. String Assertions**

| Method                          | Description                              | Syntax Example                          |
| ------------------------------- | ---------------------------------------- | --------------------------------------- |
| `assertMultiLineEqual(a, b)`    | Check if multi-line strings are equal    | `self.assertMultiLineEqual(str1, str2)` |
| `assertRegex(text, pattern)`    | Check if `pattern` matches `text`        | `self.assertRegex('abc123', r'\d+')`    |
| `assertNotRegex(text, pattern)` | Check if `pattern` does not match `text` | `self.assertNotRegex('abc', r'\d+')`    |

---

### **12. Fail Immediately**

| Method      | Description            | Syntax Example                  |
| ----------- | ---------------------- | ------------------------------- |
| `fail(msg)` | Forcefully fail a test | `self.fail("Unreachable code")` |

---
