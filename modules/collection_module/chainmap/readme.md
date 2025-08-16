# ChainMap

`ChainMap` groups multiple dictionaries (or mappings) together to be treated as a single unit.

## Syntax
```python
from collections import ChainMap
```

## Example
```python
from collections import ChainMap
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}
c = ChainMap(d1, d2)
print(c['b'])  # Output: 2 (from d1)
print(c['c'])  # Output: 4 (from d2)
```

## Features
- Searches each mapping in order
- Useful for combining multiple dicts

## Tips
- Use for layered configuration (defaults, user settings, etc.).
- Changes only affect the first mapping.
- Can add or remove mappings dynamically with `new_child()` and `parents`.
