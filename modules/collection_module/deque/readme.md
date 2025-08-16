# deque

`deque` (double-ended queue) is a list-like container with fast appends and pops from both ends.

## Syntax
```python
from collections import deque
```

## Example
```python
from collections import deque
d = deque([1, 2, 3])
d.append(4)
d.appendleft(0)
print(d)  # Output: deque([0, 1, 2, 3, 4])
d.pop()
print(d)  # Output: deque([0, 1, 2, 3])
```

## Features
- Fast O(1) operations for appends and pops from both ends
- Useful for queues and stacks

## Tips
- Use `maxlen` to create a fixed-size queue.
- Efficient for implementing BFS, sliding window, and undo features.
- Use `rotate()` to shift elements efficiently.
