## Selenium Testing in Django 

### Purpose

Used for **automated browser-based testing** (end-to-end testing) that simulates real user interactions in the UI.

---

### Setup

#### Install Selenium & WebDriver

```bash
pip install selenium
```

You also need a browser driver:

* For Chrome: [ChromeDriver](https://sites.google.com/chromium.org/driver/)
* For Firefox: [GeckoDriver](https://github.com/mozilla/geckodriver)

Ensure it's in your PATH or specify its location in the test.

---

### Django Setup for Live Testing

Django provides `StaticLiveServerTestCase` for Selenium tests:

```python
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver

class MySeleniumTests(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.driver = webdriver.Chrome()  # or Firefox, etc.

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        super().tearDownClass()

    def test_login_flow(self):
        self.driver.get(f'{self.live_server_url}/login/')
        self.driver.find_element("name", "username").send_keys("admin")
        self.driver.find_element("name", "password").send_keys("admin")
        self.driver.find_element("name", "submit").click()
        self.assertIn("Dashboard", self.driver.title)
```

---

### Common Use Cases

* Form submission
* Page redirection and routing validation
* JavaScript interaction testing
* User authentication flow
* Visual regression

---

### Best Practices

* Use test accounts and test database
* Isolate each test to avoid state dependency
* Cleanup browser state after tests
* Avoid overuse for small logic tests (use `unittest/pytest` for those)
* Combine with CI tools (GitHub Actions, Jenkins) using headless browsers

---

### Headless Mode (CI/CD Friendly)

```python
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
```

---

### Organizing Selenium Tests

```bash
myapp/
└── tests/
    ├── __init__.py
    └── selenium/
        ├── __init__.py
        ├── test_login_flow.py
        └── test_navigation.py
```

---
