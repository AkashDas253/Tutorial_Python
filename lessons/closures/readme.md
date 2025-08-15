# Closures

Closures are functions that capture variables from their enclosing scope. They are useful for creating function factories or maintaining state.

## Syntax

```python
def outer_function(parameters):
    def inner_function():
        # Access variables from outer_function
        pass
    return inner_function
```

## Example

```python
def outer_function(text):
    def inner_function():
        print(text)
    return inner_function

closure = outer_function("Hello, World!")
closure()  # Output: Hello, World!
```
