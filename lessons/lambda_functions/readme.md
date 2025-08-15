# Lambda Functions

Lambda functions are anonymous functions defined using the `lambda` keyword. They are often used for short-term tasks.

## Syntax

```python
lambda arguments: expression
```

## Example

```python
# Lambda function to add two numbers
add = lambda x, y: x + y
print(add(5, 3))  # Output: 8

# Using lambda with built-in functions
numbers = [1, 2, 3, 4]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # Output: [1, 4, 9, 16]
```
