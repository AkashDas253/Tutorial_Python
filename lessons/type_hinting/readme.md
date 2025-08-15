# Type Hinting

Type hinting allows specifying the expected data types of function arguments and return values. It improves code readability and helps with static analysis tools.

## Syntax

```python
from typing import List, Dict, Tuple

def function_name(param1: int, param2: str) -> bool:
    pass
```

## Examples

```python
from typing import List

def add_numbers(a: int, b: int) -> int:
    return a + b

def get_names() -> List[str]:
    return ["Alice", "Bob"]

print(add_numbers(5, 3))  # Output: 8
print(get_names())  # Output: ['Alice', 'Bob']
```
