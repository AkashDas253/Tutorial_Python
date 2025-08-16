## Raise Exceptions in Python

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
