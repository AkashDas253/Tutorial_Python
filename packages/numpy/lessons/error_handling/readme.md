## **Error Handling in NumPy**  

## **Overview**  
NumPy provides mechanisms to handle errors in numerical computations, such as division by zero, overflow, underflow, and invalid operations. The `numpy.seterr()` and `numpy.errstate()` functions allow control over how these errors are managed.

## **Types of Floating-Point Errors**  
| Error Type  | Cause | Example |
|-------------|------|---------|
| **Divide-by-Zero** | Division by zero | `1 / 0.0` |
| **Overflow** | Exceeding max float value | `np.exp(1000)` |
| **Underflow** | Too small to represent | `np.exp(-1000)` |
| **Invalid Operation** | Undefined operations | `np.sqrt(-1)` |

## **Controlling Error Behavior**  

### **Using `numpy.seterr()`**  
The `numpy.seterr()` function globally sets how NumPy handles floating-point errors.

#### **Modes for Handling Errors**
| Mode       | Behavior |
|------------|----------|
| `'ignore'` | Ignores the error (default for underflow) |
| `'warn'`   | Prints a warning message |
| `'raise'`  | Raises an exception (`FloatingPointError`) |
| `'call'`   | Calls a user-defined function |
| `'print'`  | Prints a warning to `stderr` |
| `'log'`    | Logs the error to a log file |

#### **Example: Setting Error Handling Globally**
```python
import numpy as np

np.seterr(divide='warn', over='raise', under='ignore', invalid='print')

# Division by zero (will warn)
print(np.array([1.0]) / 0)

# Overflow (will raise an exception)
print(np.exp(1000))  # Raises FloatingPointError

# Underflow (ignored)
print(np.exp(-1000))  # Prints '0.0'

# Invalid operation (will print)
print(np.sqrt(-1))  # Prints 'invalid value encountered in sqrt'
```

### **Using `numpy.errstate()` (Local Handling)**  
`numpy.errstate()` is a context manager that temporarily overrides error settings within a specific block.

#### **Example: Using `errstate()` Locally**
```python
with np.errstate(divide='ignore', invalid='raise'):
    print(np.array([1.0]) / 0)  # No warning
    print(np.sqrt(-1))  # Raises FloatingPointError
```
After exiting the block, NumPy reverts to the previous error settings.

## **Handling Errors with Custom Functions**  
The `'call'` mode allows defining custom error-handling functions.

#### **Example: Logging Errors**
```python
def log_errors(err, flag):
    print(f"NumPy Error: {err}, Flag: {flag}")

np.seterr(call=log_errors)

np.log(-1)  # Calls log_errors()
```

## **Suppressing Warnings Temporarily**  
Warnings can be suppressed temporarily using `np.warnings.simplefilter()`.  
```python
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

print(np.log(-1))  # No warning shown
```

## **Conclusion**  
- Use `numpy.seterr()` for **global** error handling.
- Use `numpy.errstate()` for **local** error control.
- Set appropriate error modes (`'ignore'`, `'warn'`, `'raise'`, etc.).
- Implement custom error handlers with `'call'`.
- Suppress warnings selectively with `warnings.simplefilter()`.