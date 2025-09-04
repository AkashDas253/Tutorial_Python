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
