# OrderedDict

`OrderedDict` is a dictionary subclass that remembers the order in which keys were first inserted.

## Syntax
```python
from collections import OrderedDict
```

## Example
```python
from collections import OrderedDict
d = OrderedDict()
d['a'] = 1
d['b'] = 2
d['c'] = 3
print(d)  # Output: OrderedDict([('a', 1), ('b', 2), ('c', 3)])
```

## Features
- Maintains insertion order of keys
- Useful for tasks where order matters

## Tips
- Use for implementing LRU caches and history tracking.
- Can compare equality with regular dicts, but order matters.
- Python 3.7+ regular dicts preserve order, but `OrderedDict` has extra methods like `move_to_end()`.
