## Built-in Exception Class Hierarchy

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
