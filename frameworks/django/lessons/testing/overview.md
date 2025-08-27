## Testing in Django

---

### Purpose of Testing in Django

* Ensure **code correctness** and **prevent regressions**.
* Validate **views, models, forms, URLs**, templates, and APIs.
* Supports **unit testing**, **integration testing**, and **end-to-end testing**.

---

### Test Types in Django

| Type              | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| Unit Tests        | Test individual functions/methods (models, utils).            |
| Integration Tests | Test flow between components (views, forms, models together). |
| Functional Tests  | Test the whole system (via client/browser).                   |
| Regression Tests  | Ensure fixed bugs don't reappear.                             |
| UI Tests          | Test HTML content, template rendering, navigation.            |

---

### Test Framework Used

* Based on **Python’s `unittest`**.
* Also supports **`pytest`** (third-party, preferred by some).

---

### Test Discovery Rules

* All test files must start with: `test_` or end with `_test.py`
* All test methods must start with: `test_`
* All test classes must inherit from `django.test.TestCase` or `unittest.TestCase`.

---

### Key Testing Classes

| Class                  | Use Case                         |
| ---------------------- | -------------------------------- |
| `django.test.TestCase` | Most common; includes DB support |
| `SimpleTestCase`       | No DB access (faster)            |
| `LiveServerTestCase`   | Used for Selenium/browser tests  |
| `TransactionTestCase`  | Used for testing DB rollbacks    |
| `Client`               | Simulates HTTP requests in tests |

---

### Structure of a Test File

```python
from django.test import TestCase
from .models import Item

class ItemModelTest(TestCase):
    def test_str_representation(self):
        item = Item(name="Test")
        self.assertEqual(str(item), "Test")
```

---

### Writing Tests

| Component      | How to Test                                    |
| -------------- | ---------------------------------------------- |
| Models         | Test field defaults, `__str__`, methods        |
| Views          | Use `self.client.get()`, `self.client.post()`  |
| URLs           | `reverse()` to generate and test routes        |
| Forms          | Validate `form.is_valid()` with input data     |
| Templates      | Use `assertTemplateUsed()`                     |
| Authentication | `self.client.login()`, `logout()`, permissions |
| APIs (DRF)     | Use `APIClient` from `rest_framework.test`     |

---

### Useful Assertions

| Assertion                          | Purpose                        |
| ---------------------------------- | ------------------------------ |
| `assertEqual(a, b)`                | a == b                         |
| `assertTrue(x)` / `assertFalse(x)` | x is True / False              |
| `assertContains(response, text)`   | HTML contains given text       |
| `assertRedirects(resp, url)`       | View redirects to another      |
| `assertTemplateUsed(resp, tpl)`    | Template was used in rendering |
| `assertQuerysetEqual(qs1, qs2)`    | Compare two querysets          |

---

### Running Tests

```bash
# Run all tests in project
python manage.py test

# Run tests in an app
python manage.py test myapp

# Run specific test class/method
python manage.py test myapp.tests.ItemModelTest
python manage.py test myapp.tests.ItemModelTest.test_str_representation
```

---

### Fixtures (Preloaded Data)

| Method            | Description                                  |
| ----------------- | -------------------------------------------- |
| `fixtures = []`   | Load JSON/YAML/XML data before test          |
| `setUp()`         | Add reusable test setup before each test     |
| `setUpTestData()` | Set up class-level data once for all methods |

---

### Coverage Tool

To measure test coverage:

```bash
pip install coverage
coverage run manage.py test
coverage report
coverage html  # Opens a browser report
```

---

### Best Practices

* Use `TestCase`, not `unittest.TestCase`, to get Django-specific helpers.
* Group tests by component (e.g., `test_models.py`, `test_views.py`).
* Keep tests isolated — don't rely on external state.
* Use meaningful test names and comments.
* Aim for at least 80% coverage.

---

###  Naming Conventions for Test Methods and Files

| Element            | Convention                              | Example                              |
| ------------------ | --------------------------------------- | ------------------------------------ |
| Test files         | Start with `test_`                      | `test_models.py`                     |
| Test classes       | Use `PascalCase`                        | `class UserModelTests:`              |
| Test methods       | Use `snake_case` and start with `test_` | `def test_user_creation():`          |
| Test function name | Be descriptive                          | `test_login_redirects_to_homepage()` |

> Avoid using camelCase or single-letter names for test methods.

---

### Recommended Test Folder Structure

Organize `tests` per app for scalability:

```
project/
├── app1/
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_views.py
│   │   ├── test_forms.py
│   │   └── test_urls.py
│   └── ...
├── app2/
│   └── tests/
│       └── ...
└── ...
```

You can also do a **flat test structure**, but modular is recommended for large projects.

---

### Pytest Integration Guide for Django

#### Installation

```bash
pip install pytest pytest-django
```

#### Configure `pytest.ini`

```ini
# pytest.ini
[pytest]
DJANGO_SETTINGS_MODULE = your_project.settings
python_files = tests.py test_*.py *_tests.py
```

#### Running Tests

```bash
pytest
```

#### Fixtures Support

Use Django's built-in fixtures or `pytest-django` fixtures like `client`, `db`, `admin_user`.

```python
def test_index_view(client):
    response = client.get("/")
    assert response.status_code == 200
```

#### Advantages of Using Pytest

* Simpler syntax than `unittest`
* Better failure tracebacks
* Rich plugin ecosystem
* Supports parameterized tests easily

---
