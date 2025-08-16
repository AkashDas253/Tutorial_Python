# Counter

`Counter` is a subclass of `dict` from the `collections` module for counting hashable objects.

## Syntax
```python
from collections import Counter
```

## Example
```python
from collections import Counter
lst = ['a', 'b', 'a', 'c', 'b', 'a']
c = Counter(lst)
print(c)  # Output: Counter({'a': 3, 'b': 2, 'c': 1})
```

## Features
- Counts elements in an iterable
- Returns a dictionary with elements as keys and counts as values

## Tips
- Use `Counter.most_common()` to get the most frequent elements.
- You can update counts with another iterable using `update()`.
- Useful for frequency analysis, histograms, and quick statistics.
