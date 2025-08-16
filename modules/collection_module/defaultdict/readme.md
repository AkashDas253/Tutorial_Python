# defaultdict

`defaultdict` is a subclass of `dict` that provides a default value for missing keys.

## Syntax
```python
from collections import defaultdict
```

## Example
```python
from collections import defaultdict
d = defaultdict(int)
d['a'] += 1
d['b'] += 2
print(d)  # Output: defaultdict(<class 'int'>, {'a': 1, 'b': 2})
```

## Features
- Automatically initializes missing keys
- Useful for grouping and counting

## Tips
- Use with `list`, `set`, or `int` for grouping, sets, or counting.
- Avoids `KeyError` exceptions.
- Great for building dictionaries of lists or sets in one pass.
