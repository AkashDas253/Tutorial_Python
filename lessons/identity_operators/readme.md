# Identity Operators

Identity operators are used to compare the objects, not if they are equal, but if they are actually the same object, with the same memory location.

## Operators

- `is` : Returns True if both variables are the same object
- `is not` : Returns True if both variables are not the same object

## Examples

```python
# is operator
a = [1, 2, 3]
b = a
print(a is b)  # Output: True

# is not operator
c = [1, 2, 3]
print(a is not c)  # Output: True
```
