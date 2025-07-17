## `collections` Module in Python

The `collections` module is part of Python’s **standard library** and provides **specialized container datatypes** that are alternatives to built-in types like `dict`, `list`, `set`, and `tuple`. These containers offer additional functionality or performance improvements.

---

### Core Collection Types

| Type           | Description                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| `namedtuple()` | Factory function for creating tuple subclasses with named fields.                                             |
| `deque`        | List-like container with fast appends and pops on either end.                                                 |
| `Counter`      | Dict subclass for counting hashable objects.                                                                  |
| `OrderedDict`  | Dict subclass that remembers insertion order (before Python 3.7, after that built-in `dict` preserves order). |
| `defaultdict`  | Dict subclass that returns a default value for missing keys.                                                  |
| `ChainMap`     | Groups multiple dictionaries into one view.                                                                   |
| `UserDict`     | Wrapper around dictionary objects for easier subclassing.                                                     |
| `UserList`     | Wrapper around list objects for easier subclassing.                                                           |
| `UserString`   | Wrapper around string objects for easier subclassing.                                                         |

---

### Detailed Overview with Key Parameters and Behavior

#### `namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)`

* Creates immutable, indexable, and named field tuples.
* Parameters:

  * `typename`: Name of the new class.
  * `field_names`: List/tuple or string of field names.
  * `rename`: If `True`, invalid names are automatically replaced.
  * `defaults`: Default values for fields (tuple).
  * `module`: Module name to assign to the class.

#### `deque([iterable[, maxlen]])`

* Doubly-ended queue.
* Fast O(1) append/pop from either end.
* Parameters:

  * `iterable`: Initial data.
  * `maxlen`: Maximum length; older items are removed once full.

#### `Counter([iterable-or-mapping])`

* Automatically counts item occurrences.
* Behaves like a dictionary.
* Methods: `.elements()`, `.most_common([n])`, `.subtract([iterable])`

#### `OrderedDict([items])`

* Preserves the insertion order.
* Has methods like `.move_to_end(key, last=True)` to reorder keys.

#### `defaultdict(default_factory)`

* Provides default values if a key is missing.
* `default_factory`: Callable that returns default (e.g., `int`, `list`).

#### `ChainMap(*maps)`

* Combines multiple dicts.
* Searches keys in order, updates only the first dict.

#### `UserDict`, `UserList`, `UserString`

* Useful for inheritance and method overriding.
* Wrapped in a class rather than using built-in directly.

---

### Internal Behaviors and Considerations

* `namedtuple` fields are immutable, but `_replace()` allows creating a modified copy.
* `deque` is optimized for append/pop operations compared to `list`.
* `Counter` supports arithmetic operations like `+`, `-`, `&`, `|` on multisets.
* `defaultdict` only creates default when key is missing.
* `OrderedDict` supports reversed iteration and equality comparison by order.

---

### Usage Scenarios

| Use Case                         | Collection Type                      |
| -------------------------------- | ------------------------------------ |
| Count elements                   | `Counter`                            |
| Track recent items (FIFO/LIFO)   | `deque`                              |
| Return default on missing key    | `defaultdict`                        |
| Maintain insertion order         | `OrderedDict`                        |
| Combine multiple scopes/configs  | `ChainMap`                           |
| Named records with field access  | `namedtuple`                         |
| Subclassing dict/list/str safely | `UserDict`, `UserList`, `UserString` |

---
