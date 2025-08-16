## Custom Exceptions in Python

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