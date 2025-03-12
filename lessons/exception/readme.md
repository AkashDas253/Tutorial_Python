## **Exception and Exception Handling in Python**

### **Definition**  
- **Exception**: An error that occurs during program execution, disrupting the normal flow.  
- **Exception Handling**: Mechanism to handle errors gracefully without crashing the program.  

---

### **1. Types of Errors**  
| Type | Description | Example |
|------|------------|---------|
| **Syntax Error** | Invalid syntax in code | `if True print("Hello")` |
| **Runtime Error (Exception)** | Error occurs during execution | `1 / 0` (ZeroDivisionError) |
| **Logical Error** | Produces incorrect results but no error | Using `+` instead of `-` |

---

### **2. Common Built-in Exceptions**  
| Exception | Cause |
|-----------|-------|
| `ZeroDivisionError` | Division by zero (`1 / 0`) |
| `TypeError` | Unsupported operation between types (`"2" + 2`) |
| `ValueError` | Invalid value (`int("abc")`) |
| `IndexError` | Out-of-range list index (`lst[10]`) |
| `KeyError` | Accessing a non-existent dictionary key (`dict['key']`) |
| `AttributeError` | Calling an undefined attribute/method (`None.length()`) |
| `NameError` | Using an undefined variable (`print(x)`, where `x` is not defined) |
| `FileNotFoundError` | File does not exist (`open("file.txt")`) |

---

### **3. Exception Handling (`try-except`)**  
- Prevents program from crashing due to an error.  

```python
try:
    x = 1 / 0  # Causes ZeroDivisionError
except ZeroDivisionError:
    print("Cannot divide by zero.")  
```

**Output:**  
`Cannot divide by zero.`  

---

### **4. Handling Multiple Exceptions**  
- Multiple `except` blocks for different errors.  

```python
try:
    num = int(input("Enter a number: "))  # Causes ValueError if input is not a number
    result = 10 / num  # Causes ZeroDivisionError if num = 0
except ValueError:
    print("Invalid number entered.")
except ZeroDivisionError:
    print("Cannot divide by zero.")
```

---

### **5. Catching All Exceptions (`Exception`)**  
- Using `except Exception` to handle all errors.  

```python
try:
    print(10 / 0)  # Causes ZeroDivisionError
except Exception as e:
    print(f"Error: {e}")
```

---

### **6. `else` and `finally` Blocks**  
| Block | Purpose |
|-------|---------|
| `else` | Executes if `try` succeeds |
| `finally` | Executes **always**, whether exception occurs or not |

```python
try:
    x = 5 / 1  # No error
except ZeroDivisionError:
    print("Cannot divide by zero.")
else:
    print("Division successful.")  # Runs if no exception
finally:
    print("Execution completed.")  # Runs always
```

**Output:**  
```
Division successful.
Execution completed.
```

---

### **7. Raising Exceptions (`raise`)**  
- Manually raising exceptions using `raise`.  

```python
age = -5
if age < 0:
    raise ValueError("Age cannot be negative.")
```

---

### **8. Custom Exceptions**  
- Creating user-defined exceptions by inheriting `Exception` class.  

```python
class NegativeAgeError(Exception):
    pass

try:
    age = int(input("Enter age: "))
    if age < 0:
        raise NegativeAgeError("Age cannot be negative.")
except NegativeAgeError as e:
    print(f"Error: {e}")
```

---

---
---

## Exception Handling

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