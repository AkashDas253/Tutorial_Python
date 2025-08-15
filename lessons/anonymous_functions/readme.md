# Anonymous Functions

Anonymous functions are functions without a name, often used for short-term tasks. In Python, they are created using the `lambda` keyword.

## Syntax

```python
lambda arguments: expression
```

## Example

```python
# Using lambda
square = lambda x: x**2
print(square(5))  # Output: 25

# Using lambda with filter
numbers = [1, 2, 3, 4]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # Output: [2, 4]
```
