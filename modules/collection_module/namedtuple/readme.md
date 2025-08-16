# namedtuple

`namedtuple` is a factory function for creating tuple subclasses with named fields.

## Syntax
```python
from collections import namedtuple
```

## Example
```python
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)  # Output: 1 2
```

## Features
- Access tuple elements by name
- Improves code readability

## Tips
- Use for lightweight objects where immutability is desired.
- Supports default values and type hints.
- Can convert to dict with `_asdict()` and replace values with `_replace()`.
