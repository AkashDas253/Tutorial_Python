## Python Built-in Functions

Python has a set of built-in functions.


### Build-in Functions:

| Function               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `abs(x)`               | Returns the absolute value of a number                                      |
| `all(iterable)`        | Returns True if all items in an iterable object are true                    |
| `any(iterable)`        | Returns True if any item in an iterable object is true                      |
| `ascii(object)`        | Returns a readable version of an object. Replaces non-ascii characters with escape character |
| `bin(x)`               | Returns the binary version of a number                                      |
| `bool([x])`            | Returns the boolean value of the specified object                           |
| `bytearray([source[, encoding[, errors]]])` | Returns an array of bytes                                                   |
| `bytes([source[, encoding[, errors]]])`     | Returns a bytes object                                                      |
| `callable(object)`     | Returns True if the specified object is callable, otherwise False           |
| `chr(i)`               | Returns a character from the specified Unicode code                         |
| `classmethod(function)`| Converts a method into a class method                                       |
| `compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)` | Returns the specified source as an object, ready to be executed             |
| `complex([real[, imag]])` | Returns a complex number                                                    |
| `delattr(object, name)`| Deletes the specified attribute (property or method) from the specified object |
| `dict(**kwargs)`       | Returns a dictionary (Array)                                                |
| `dir([object])`        | Returns a list of the specified object's properties and methods             |
| `divmod(a, b)`         | Returns the quotient and the remainder when argument1 is divided by argument2 |
| `enumerate(iterable, start=0)` | Takes a collection (e.g. a tuple) and returns it as an enumerate object     |
| `eval(expression, globals=None, locals=None)` | Evaluates and executes an expression                                        |
| `exec(object[, globals[, locals]])` | Executes the specified code (or object)                                     |
| `filter(function, iterable)` | Use a filter function to exclude items in an iterable object                |
| `float([x])`           | Returns a floating point number                                             |
| `format(value[, format_spec])` | Formats a specified value                                                   |
| `frozenset([iterable])` | Returns a frozenset object                                                  |
| `getattr(object, name[, default])` | Returns the value of the specified attribute (property or method)           |
| `globals()`            | Returns the current global symbol table as a dictionary                     |
| `hasattr(object, name)`| Returns True if the specified object has the specified attribute (property/method) |
| `hash(object)`         | Returns the hash value of a specified object                                |
| `help([object])`       | Executes the built-in help system                                           |
| `hex(x)`               | Converts a number into a hexadecimal value                                  |
| `id(object)`           | Returns the id of an object                                                 |
| `input([prompt])`      | Allowing user input                                                         |
| `int([x[, base]])`     | Returns an integer number                                                   |
| `isinstance(object, classinfo)` | Returns True if a specified object is an instance of a specified object     |
| `issubclass(class, classinfo)` | Returns True if a specified class is a subclass of a specified object       |
| `iter(object[, sentinel])` | Returns an iterator object                                                  |
| `len(s)`               | Returns the length of an object                                             |
| `list([iterable])`     | Returns a list                                                              |
| `locals()`             | Returns an updated dictionary of the current local symbol table             |
| `map(function, iterable, ...)` | Returns the specified iterator with the specified function applied to each item |
| `max(iterable, *[, key, default])` | Returns the largest item in an iterable                                     |
| `memoryview(obj)`      | Returns a memory view object                                                |
| `min(iterable, *[, key, default])` | Returns the smallest item in an iterable                                    |
| `next(iterator[, default])` | Returns the next item in an iterable                                        |
| `object()`             | Returns a new object                                                        |
| `oct(x)`               | Converts a number into an octal                                             |
| `open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)` | Opens a file and returns a file object                                      |
| `ord(c)`               | Convert an integer representing the Unicode of the specified character      |
| `pow(x, y[, z])`       | Returns the value of x to the power of y                                    |
| `print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)` | Prints to the standard output device                                        |
| `property(fget=None, fset=None, fdel=None, doc=None)` | Gets, sets, deletes a property                                              |
| `range(stop)` or `range(start, stop[, step])` | Returns a sequence of numbers, starting from 0 and increments by 1 (by default) |
| `repr(object)`         | Returns a readable version of an object                                     |
| `reversed(seq)`        | Returns a reversed iterator                                                 |
| `round(number[, ndigits])` | Rounds a number                                                             |
| `set([iterable])`      | Returns a new set object                                                    |
| `setattr(object, name, value)` | Sets an attribute (property/method) of an object                            |
| `slice(stop)` or `slice(start, stop[, step])` | Returns a slice object                                                      |
| `sorted(iterable, *, key=None, reverse=False)` | Returns a sorted list                                                       |
| `staticmethod(function)` | Converts a method into a static method                                     |
| `str([object[, encoding[, errors]]])` | Returns a string object                                                     |
| `sum(iterable[, start])` | Sums the items of an iterator                                               |
| `super([type[, object-or-type]])` | Returns an object that represents the parent class                          |
| `tuple([iterable])`    | Returns a tuple                                                             |
| `type(object)` or `type(name, bases, dict)` | Returns the type of an object                                               |
| `vars([object])`       | Returns the `__dict__` property of an object                                |
| `zip(*iterables)`      | Returns an iterator, from two or more iterators                             |

