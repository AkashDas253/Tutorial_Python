# Built-in Higher Order Functions

Higher-order functions take other functions as arguments or return them. Common examples include `map`, `filter`, and `reduce`.

## Syntax

```python
# map
map(function, iterable)

# filter
filter(function, iterable)

# reduce
from functools import reduce
reduce(function, iterable)
```

## Examples

```python
from functools import reduce

# map()
numbers = [1, 2, 3, 4]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # Output: [1, 4, 9, 16]

# filter()
even = filter(lambda x: x % 2 == 0, numbers)
print(list(even))  # Output: [2, 4]

# reduce()
sum = reduce(lambda x, y: x + y, numbers)
print(sum)  # Output: 10
```
