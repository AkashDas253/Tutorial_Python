## Logging and Output in Pytest

### Purpose

* Pytest provides a flexible system for managing logs and controlling the output generated during test execution.
* This allows users to easily debug tests and configure how results are displayed or saved.

### Log Capture

Pytest has built-in support for capturing log output during test execution, which can be configured to show logs in various formats.

#### Enabling Log Capture

* Pytest captures log messages generated during test runs, and these messages are suppressed by default unless specifically requested.
* To capture logs in the output, use the `-s` option to disable output capturing.

```bash
pytest -s
```

This will show print statements and logs directly in the console.

#### Capturing Logs for Specific Levels

To capture logs for specific log levels, use the `--log-level` option:

```bash
pytest --log-level=DEBUG
```

This will display logs for messages with a level of `DEBUG` or higher (e.g., `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

#### Customizing Log Output Format

You can specify the format of the log output using `--log-format`. The default format is:

```
%(asctime)s %(levelname)-8s %(message)s
```

You can customize it by specifying a different format:

```bash
pytest --log-format="%(levelname)s: %(message)s"
```

#### Storing Logs to a File

To store logs in a file, use the `--log-file` option:

```bash
pytest --log-file=logfile.log
```

This will save all captured logs to `logfile.log`. You can combine this with other logging options, like `--log-level` and `--log-format`, to further control the logging behavior.

### Logging in Test Code

Pytest supports the integration of Python’s `logging` module for logging messages during test execution.

#### Example of Logging in Test Code

```python
import logging

def test_example():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("This is a debug message")
    assert True
```

This will output the log message during the test run.

#### Configuring Logging in `pytest.ini`

You can configure the default logging level in the `pytest.ini` file.

```ini
[pytest]
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s - %(name)s - %(levelname)s - %(message)s
```

* **log\_cli**: Enables logging to the console.
* **log\_cli\_level**: Sets the logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
* **log\_cli\_format**: Specifies the log format.

### Controlling Output Verbosity

#### Default Output

Pytest outputs detailed information about each test, showing a dot (`.`) for passed tests, `F` for failed tests, and `E` for errors.

#### Verbose Mode (`-v`)

Use the `-v` flag to get more detailed output, including the name of each test and its outcome:

```bash
pytest -v
```

#### Quiet Mode (`-q`)

To suppress most of the output and only display essential information, use the `-q` (quiet) option:

```bash
pytest -q
```

This is especially useful for large test suites when you only need to know whether the tests passed or failed.

### Test Result Summary

Pytest provides a summary of test results, including the number of passed, failed, skipped, and other statuses after the test run.

Example of summary output:

```bash
================================== test session starts ================================
collected 10 items

test_example.py::test_example PASSED
test_example.py::test_example_fail FAILED
test_example.py::test_example_skip SKIPPED

================================ 1 failed, 1 skipped, 8 passed in 1.23 seconds ===================================
```

### Failures and Errors

By default, Pytest shows a traceback when tests fail, which helps in identifying the exact cause.

#### Customizing Traceback Style

You can configure the traceback style using the `--tb` option. Options include:

* `short`: Only shows the last few lines of the traceback.
* `long`: Shows the full traceback, which is useful for debugging.
* `line`: Displays the traceback as a single line.

Example:

```bash
pytest --tb=short
```

### Controlling Test Output with `pytest.ini`

You can configure output-related settings in `pytest.ini` to control verbosity and logging.

#### Example of `pytest.ini`

```ini
[pytest]
addopts = --maxfail=3 --disable-warnings --tb=short
log_cli = true
log_cli_level = DEBUG
log_cli_format = %(asctime)s - %(levelname)s - %(message)s
```

* **addopts**: Automatically applies additional options like limiting the number of failures and disabling warnings.
* **log\_cli**: Enables log output to the console.
* **log\_cli\_level**: Sets the log level for the console logs.
* **log\_cli\_format**: Specifies the log output format.

### Capturing Output from External Programs

You can capture the output from external programs or subprocesses used within your tests with `capsys` or `capfd` fixtures.

#### Using `capsys` Fixture

```python
def test_print_output(capsys):
    print("Hello, World!")
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
```

#### Using `capfd` Fixture (for file descriptors)

```python
def test_fd_output(capfd):
    import sys
    sys.stdout.write("Hello, World!")
    captured = capfd.readouterr()
    assert captured.out == "Hello, World!"
```

### Test Reporting

For more advanced logging and output, Pytest can be integrated with third-party plugins like `pytest-html`, `pytest-json`, and others to generate HTML or JSON reports.

#### Example of HTML Report

```bash
pytest --html=report.html
```

This generates an HTML report of the test results that you can easily share with others.

---
