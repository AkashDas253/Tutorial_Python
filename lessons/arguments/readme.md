# Arguments

Arguments are values passed to functions when they are called. They allow functions to accept input and produce dynamic results.

## Syntax

```python
def function_name(positional_arg, keyword_arg=value, *args, **kwargs):
    # Function body
    pass
```

## Types of Arguments

1. **Positional Arguments**: Passed in order.
2. **Keyword Arguments**: Passed with parameter names.
3. **Default Arguments**: Have default values.
4. **Variable-length Arguments**:
   - `*args`: For tuples.
   - `**kwargs`: For dictionaries.

## Example

```python
def greet(name, age=30, *args, **kwargs):
    print(f"Hello, {name}. You are {age} years old.")
    print("Additional args:", args)
    print("Additional kwargs:", kwargs)

# Calling the function
greet("Alice", 25, "extra", key="value")
```
