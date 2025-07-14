### Exception Handling

- When an error occurs, Python will normally stop and generate an error message.
- These exceptions can be handled using the `try` statement.

- `try`: The block of code to be tested for exceptions.
- `except ExceptionClass_Name as e`: The block of code to be executed if a specific exception occurs. The exception instance is referred to as `e`.
- `except`: The block of code to be executed if any exception occurs.
- `else`: The block of code to be executed if no exceptions occur.
- `finally`: The block of code to be executed regardless of whether an exception occurs or not.

#### Syntax

```python
try:
    # code that may raise an exception
    pass

except ExceptionClass_Name as e:
    # code to be executed in case of specific type of exception
    # refer to exception using 'e'
    pass

except:
    # code that runs if an exception occurs
    pass

else:
    # code to be executed if no exception occurs
    pass

finally:
    # code that runs no matter what (exception or not)
    pass

```

### Raise Exceptions in Python

- To throw (raise) an exception in Python, you use the `raise` statement followed by an instance of the exception class.

#### Basic Syntax

```python
raise ExceptionType("Error message")
```

### Common Built-in Exceptions in Python

| Exception Class       | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `Exception`           | Base class for all exceptions.                                              |
| `AttributeError`      | Raised when an attribute reference or assignment fails.                     |
| `EOFError`            | Raised when the `input()` function hits an end-of-file condition.           |
| `ImportError`         | Raised when an import statement fails to find the module definition.        |
| `IndexError`          | Raised when a sequence subscript is out of range.                           |
| `KeyError`            | Raised when a dictionary key is not found.                                  |
| `KeyboardInterrupt`   | Raised when the user hits the interrupt key (usually `Ctrl+C` or `Delete`). |
| `MemoryError`         | Raised when an operation runs out of memory.                                |
| `NameError`           | Raised when a local or global name is not found.                            |
| `OSError`             | Raised when a system-related operation causes an error.                     |
| `OverflowError`       | Raised when the result of an arithmetic operation is too large to be expressed. |
| `RuntimeError`        | Raised when an error is detected that doesn't fall in any of the other categories. |
| `StopIteration`       | Raised by the `next()` function to indicate that there are no further items produced by the iterator. |
| `SyntaxError`         | Raised when the parser encounters a syntax error.                           |
| `IndentationError`    | Raised when there is incorrect indentation.                                 |
| `TabError`            | Raised when indentation consists of inconsistent tabs and spaces.           |
| `SystemError`         | Raised when the interpreter finds an internal error.                        |
| `TypeError`           | Raised when an operation or function is applied to an object of inappropriate type. |
| `UnboundLocalError`   | Raised when a local variable is referenced before it has been assigned.     |
| `ValueError`          | Raised when a function receives an argument of the correct type but inappropriate value. |
| `ZeroDivisionError`   | Raised when the second argument of a division or modulo operation is zero.  |

### Built-in Exception Class Hierarchy

- `BaseException`
  - The base class for all built-in exceptions. It is not meant to be directly inherited by user-defined classes.

  - `SystemExit`
    - Raised by the `sys.exit()` function.
  
  - `KeyboardInterrupt`
    - Raised when the user interrupts program execution, usually by pressing `Ctrl+C`.

  - `GeneratorExit`
    - Raised when a generator or coroutine is closed.

  - `Exception`
    - The base class for all built-in, non-system-exiting exceptions.

    - `StopIteration`
      - Raised by the `next()` function to indicate that there are no further items produced by the iterator.

    - `ArithmeticError`
      - The base class for all errors that occur for numeric calculations.

      - `FloatingPointError`
        - Raised when a floating point operation fails.

      - `OverflowError`
        - Raised when the result of an arithmetic operation is too large to be expressed.

      - `ZeroDivisionError`
        - Raised when the second argument of a division or modulo operation is zero.

    - `AssertionError`
      - Raised when an `assert` statement fails.

    - `AttributeError`
      - Raised when an attribute reference or assignment fails.

    - `BufferError`
      - Raised when a buffer-related operation cannot be performed.

    - `EOFError`
      - Raised when the `input()` function hits an end-of-file condition.

    - `ImportError`
      - Raised when an import statement fails to find the module definition.

      - `ModuleNotFoundError`
        - Raised when a module could not be found.

    - `LookupError`
      - The base class for all lookup errors.

      - `IndexError`
        - Raised when a sequence subscript is out of range.

      - `KeyError`
        - Raised when a dictionary key is not found.

    - `MemoryError`
      - Raised when an operation runs out of memory.

    - `NameError`
      - Raised when a local or global name is not found.

      - `UnboundLocalError`
        - Raised when a local variable is referenced before it has been assigned.

    - `OSError`
      - Raised when a system-related operation causes an error.

      - `BlockingIOError`
        - Raised when an operation would block on an object (e.g., socket) set for non-blocking operation.

      - `ChildProcessError`
        - Raised when an operation on a child process fails.

      - `ConnectionError`
        - The base class for connection-related errors.

        - `BrokenPipeError`
          - Raised when a pipe is broken.

        - `ConnectionAbortedError`
          - Raised when a connection attempt is aborted.

        - `ConnectionRefusedError`
          - Raised when a connection attempt is refused.

        - `ConnectionResetError`
          - Raised when a connection is reset.

      - `FileExistsError`
        - Raised when trying to create a file or directory which already exists.

      - `FileNotFoundError`
        - Raised when a file or directory is requested but doesn't exist.

      - `InterruptedError`
        - Raised when a system call is interrupted by an incoming signal.

      - `IsADirectoryError`
        - Raised when a file operation is requested on a directory.

      - `NotADirectoryError`
        - Raised when a directory operation is requested on something which is not a directory.

      - `PermissionError`
        - Raised when trying to run an operation without the adequate access rights.

      - `ProcessLookupError`
        - Raised when a given process is not found.

      - `TimeoutError`
        - Raised when a system function times out at the system level.

    - `ReferenceError`
      - Raised when a weak reference proxy is used to access a garbage collected referent.

    - `RuntimeError`
      - Raised when an error is detected that doesn't fall in any of the other categories.

      - `NotImplementedError`
        - Raised when an abstract method that needs to be implemented in an inherited class is not actually implemented.

      - `RecursionError`
        - Raised when the maximum recursion depth is exceeded.

    - `SyntaxError`
      - Raised when the parser encounters a syntax error.

      - `IndentationError`
        - Raised when there is incorrect indentation.

        - `TabError`
          - Raised when indentation consists of inconsistent tabs and spaces.

    - `SystemError`
      - Raised when the interpreter finds an internal error.

    - `TypeError`
      - Raised when an operation or function is applied to an object of inappropriate type.

    - `ValueError`
      - Raised when a function receives an argument of the correct type but inappropriate value.

      - `UnicodeError`
        - Raised when a Unicode-related encoding or decoding error occurs.

        - `UnicodeDecodeError`
          - Raised when a Unicode-related decoding error occurs.

        - `UnicodeEncodeError`
          - Raised when a Unicode-related encoding error occurs.

        - `UnicodeTranslateError`
          - Raised when a Unicode-related translation error occurs.

    - `Warning`
      - The base class for all warning categories.

      - `DeprecationWarning`
        - Raised for deprecated features.

      - `PendingDeprecationWarning`
        - Raised for features that are deprecated but not yet removed.

      - `RuntimeWarning`
        - Raised for runtime warnings.

      - `SyntaxWarning`
        - Raised for syntax warnings.

      - `UserWarning`
        - Raised for user-defined warnings.

      - `FutureWarning`
        - Raised for warnings about changes in the future.

      - `ImportWarning`
        - Raised for warnings related to module imports.

      - `UnicodeWarning`
        - Raised for warnings related to Unicode.

      - `BytesWarning`
        - Raised for warnings related to bytes and bytearray.

      - `ResourceWarning`
        - Raised for warnings related to resource usage.

### Custom Exceptions in Python

You can create custom exceptions in Python by defining a new class that inherits from the built-in `Exception` class or any other `built-in exception` class.

#### Steps to Create a Custom Exception

1. Define a new class that inherits from `Exception`.
2. Optionally, override the `__init__` method to accept custom arguments.
3. Optionally, override the `__str__` or `__repr__` methods to customize the string representation.

#### Special Attributes of Exception Objects

- `args`
  - A tuple of arguments given to the exception constructor. This is typically used to store the error message.
  - Example:
    ```python
    try:
        raise ValueError("Invalid value")
    except ValueError as e:
        print(e.args)  # Output: ('Invalid value',)
    ```

- `__str__()`
  - Returns the string representation of the exception, typically the error message.
  - Example:
    ```python
    try:
        raise ValueError("Invalid value")
    except ValueError as e:
        print(str(e))  # Output: Invalid value
    ```

- `__repr__()`
  - Returns the official string representation of the exception, which can be used to recreate the exception object.
  - Example:
    ```python
    try:
        raise ValueError("Invalid value")
    except ValueError as e:
        print(repr(e))  # Output: ValueError('Invalid value')
    ```

### Custom Exceptions

#### Syntax

```python
class CustomError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(message)

    def __str__(self):
        return f"[Error {self.code}]: {self.message}"

# Usage
try:
    raise CustomError("Something went wrong", 404)
except CustomError as e:
    print("args:", e.args)          # Output: args: ('Something went wrong',)
    print("str:", str(e))           # Output: str: [Error 404]: Something went wrong
    print("repr:", repr(e))         # Output: repr: CustomError('Something went wrong')
    print("message:", e.message)    # Output: message: Something went wrong
    print("code:", e.code)          # Output: code: 404

``` 