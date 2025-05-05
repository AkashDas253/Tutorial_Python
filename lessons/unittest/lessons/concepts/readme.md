
### pytest Concepts and Subconcepts

- **Test Discovery**
  - File Naming Conventions (`test_*.py`, `*_test.py`)
  - Test Function Naming Conventions (`test_`)
  
- **Test Execution**
  - Running Tests (`pytest`)
  - Running Specific Tests
  - Output Formats (Default, Detailed, XML)
  
- **Assertions**
  - Using `assert` Statements
  - Detailed Assertion Failures
  - Assertion introspection
  
- **Fixtures**
  - Defining Fixtures (`@pytest.fixture`)
  - Fixture Scope (Function, Class, Module, Session)
  - Yield-Based Fixtures (Setup and Teardown)
  - Autouse Fixtures
  - Fixture Dependencies
  
- **Markers**
  - Marking Tests (`@pytest.mark`)
  - Custom Markers
  - Marker-based Test Selection (e.g., `pytest -k`)

- **Test Parametrization**
  - Parametrize Decorator (`@pytest.mark.parametrize`)
  - Multiple Parameterization
  - Combining Multiple Parametrized Tests

- **Test Skipping and Expected Failures**
  - Skipping Tests (`@pytest.mark.skip`)
  - Conditional Skipping (`@pytest.mark.skipif`)
  - Expected Failures (`@pytest.mark.xfail`)
  - Skipping Tests Based on Conditions
  
- **Test Reporting**
  - Verbose Output
  - Test Result Reporting (Success, Fail, Skip)
  - JUnit XML Output
  - Reporting Plugins (e.g., `pytest-html`)
  
- **Plugins**
  - Built-in Plugins (e.g., `pytest-cov`, `pytest-xdist`)
  - Custom Plugins
  - Plugin Configuration and Usage
  
- **Test Setup and Teardown**
  - Setup and Teardown Using Fixtures
  - Module and Class-Level Setup
  - Setup and Teardown with `yield` in Fixtures
  
- **Test Dependencies**
  - Sharing Data Between Tests Using Fixtures
  - Fixture Injection into Test Functions
  - Scoped Fixtures for Cross-Test Dependencies
  
- **Parallel Test Execution**
  - Parallel Execution with `pytest-xdist`
  - Load Balancing Across Test Workers
  - Running Tests in Parallel (`pytest -n`)
  
- **Mocking and Patching**
  - Using `unittest.mock` with pytest
  - Mocking External APIs and Services
  - Patching Functions and Classes in Tests
  
- **Test Coverage**
  - Measuring Test Coverage with `pytest-cov`
  - Coverage Reporting (`--cov`)
  - Combining Coverage Reports
  
- **Test Organization**
  - Organizing Tests into Files and Directories
  - Using Test Suites and Subdirectories
  - Grouping Tests with Markers
  
- **Test Fixtures Usage**
  - Function-Level Fixtures
  - Class-Level Fixtures
  - Module-Level Fixtures
  - Session-Level Fixtures
  - Autouse Fixtures for Implicit Use
  
- **Test Assertions and Fixtures Interaction**
  - Parametrized Fixtures
  - Using Fixtures for Cleanup After Tests
  - Passing Fixtures as Arguments in Test Functions
  
- **Advanced Test Selection**
  - Running Tests with Specific Markers (`-m`)
  - Selecting Tests by Name (`-k`)
  - Running Tests Based on Conditions
  
- **Test Debugging**
  - Debugging Failing Tests with `pytest --pdb`
  - Using `pytest.set_trace()` for Interactive Debugging
  
- **Test Results Handling**
  - Handling Test Failures with Hooks
  - Customizing Failure Output
  
- **Mocking External Systems**
  - Mocking Functions with `pytest-mock`
  - Patching APIs, Database Calls, and Services

---