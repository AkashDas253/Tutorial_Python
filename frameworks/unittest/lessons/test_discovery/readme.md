## Test Discovery in `unittest`

---

#### **Purpose**

Test discovery allows `unittest` to **automatically locate and run test cases** across files and directories without manually specifying each test.

---

### **1. Discovery Requirements**

* **Test file names** should match the pattern:
  `test*.py` (default pattern)

* **Test class requirements**:

  * Must inherit from `unittest.TestCase`
  * Should not have an `__init__` method

* **Test method requirements**:

  * Method names must start with `test_`

---

### **2. Discovery via Command Line**

Run test discovery from the project root or desired directory:

```bash
python -m unittest discover
```

**Custom options**:

| Option               | Description                                       |
| -------------------- | ------------------------------------------------- |
| `-s <start_dir>`     | Directory to start discovery (default: `.`)       |
| `-p <pattern>`       | Pattern to match test files (default: `test*.py`) |
| `-t <top_level_dir>` | Top-level directory of project (optional)         |

**Example**:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

### **3. Discovery via Python Script**

You can discover tests programmatically using `unittest.TestLoader.discover()`.

**Syntax**:

```python
import unittest

# Create a test suite via discovery
suite = unittest.TestLoader().discover(
    start_dir='tests',     # folder to start from
    pattern='test_*.py',   # file pattern
    top_level_dir=None     # defaults to current directory
)

# Run the suite
unittest.TextTestRunner(verbosity=2).run(suite)
```

---

### **4. Directory Structure Recommendation**

```text
project/
├── module/
│   └── logic.py
└── tests/
    ├── __init__.py
    ├── test_logic.py
    └── test_utils.py
```

Use `__init__.py` to make `tests` a package if importing modules.

---

### **5. Mixing Manual and Discovered Tests**

You can combine manually added tests with discovered ones in a `TestSuite`:

```python
manual_suite = unittest.TestSuite()
manual_suite.addTest(SomeTestClass('test_method'))

discovered_suite = unittest.TestLoader().discover('tests')
combined_suite = unittest.TestSuite([manual_suite, discovered_suite])

unittest.TextTestRunner().run(combined_suite)
```

---

### **6. Limitations & Tips**

* Discovery ignores files/classes/methods that don’t follow naming rules.
* For custom loaders, use subclassing from `unittest.TestLoader`.
* Do not name your test scripts the same as Python standard modules (e.g., `unittest.py`).

---
