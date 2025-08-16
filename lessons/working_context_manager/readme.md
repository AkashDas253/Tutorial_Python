# How Context Managers Work Under the Hood

A **context manager** controls what happens **when entering and exiting a block** of code (the `with` statement).

### Flow of Execution

1. **`with <context_manager> as <variable>`**

   * Python calls the **`__enter__()`** method of the context manager object.
   * The value returned by `__enter__()` is assigned to the variable (optional).

2. **Code inside the `with` block executes**

   * Any statements inside the block are executed.
   * If an exception occurs inside the block, Python **records the exception type, value, and traceback**.

3. **`__exit__(exc_type, exc_value, traceback)` is called**

   * Always called, whether the block completes normally or raises an exception.
   * Parameters:

     * `exc_type` → Type of exception (e.g., `ZeroDivisionError`), `None` if no exception.
     * `exc_value` → The exception instance.
     * `traceback` → Traceback object.
   * Cleanup code is executed here (e.g., closing files, releasing locks).

4. **Exception Handling**

   * If `__exit__` returns **`True`**, the exception is **suppressed**.
   * If it returns **`False`** or nothing, the exception is **propagated** after cleanup.

---

## Diagram of Context Manager Flow

```text
Start 'with' statement
          │
          ▼
    Call __enter__()
          │
          ▼
   Execute block code
          │
      Exception?
      ┌─────────────┐
      │ Yes         │
      ▼             │
  Pass exc_type, exc_value, traceback
          │
    Call __exit__(exc_type, exc_value, traceback)
          │
Return True? ──► Exception suppressed
     │No
     ▼
Exception propagated after cleanup
```

---

## Example – Normal Execution (No Exception)

```python
class MyContext:
    def __enter__(self):
        print("Enter block")
        return "Resource"
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exit block")
        return False  # Do not suppress exceptions

with MyContext() as res:
    print("Using", res)
```

**Output:**

```
Enter block
Using Resource
Exit block
```

---

## Example – With Exception Handling

```python
class SafeDivide:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Handled exception: {exc_value}")
            return True  # Suppress exception

with SafeDivide() as sd:
    result = 10 / 0  # ZeroDivisionError occurs
print("Program continues")  # Execution continues
```

**Output:**

```
Handled exception: division by zero
Program continues
```

* `ZeroDivisionError` occurred inside the block.
* `__exit__` received the exception details.
* Returning `True` **suppressed the exception**, so the program continues.

---

### Key Takeaways

* **`__enter__()`** → Resource acquisition or setup.
* **`__exit__()`** → Resource cleanup + optional exception handling.
* **Exceptions are propagated** unless `__exit__` explicitly returns `True`.
* Python guarantees **deterministic cleanup** even with exceptions.

---
