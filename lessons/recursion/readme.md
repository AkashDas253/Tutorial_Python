# Recursion

Recursion occurs when a function calls itself. It is useful for solving problems that can be broken down into smaller, similar problems.

## Syntax

```python
def recursive_function(parameters):
    if base_condition:
        return value
    return recursive_function(modified_parameters)
```

## Example

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # Output: 120
```
