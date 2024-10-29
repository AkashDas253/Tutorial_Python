# Basics

## Syntax:

### Execute Python Syntax:       

    - `python myfile.py` in terminal
    - code snippets in cell
### Indentation:  

- Code at same block at same level
### Variable creation:

 `variable_name = value`

### Comments: 
```py
# single-line comment

"""  
-- Multiline comment -- 
line 1 
line 2 \n line 3"""

```  

## Variables:

### Overview

- Variables are containers for storing data values.
- Created by assigning a value to a name.

### Creating Variables

- No command for declaring a variable.
- Created upon assignment.

#### Example
```python
x = 5
y = "John"
print(x)
print(y)
```

### Dynamic Typing

- Variables do not need a specific type declaration.
- Can change type after assignment.

#### Example
```python
x = 4       # x is of type int
x = "Sally" # x is now of type str
print(x)
```

### Casting

- Specify data type using casting.

#### Example
```python
x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0
```

### Get the Type

- Use `type()` function to get the data type.

#### Example
```python
x = 5
y = "John"
print(type(x))
print(type(y))
```

### String Quotes

- Strings can be declared with single or double quotes.

#### Example
```python
x = "John"
# is the same as
x = 'John'
```

### Case Sensitivity

- Variable names are case-sensitive.

#### Example
```python
a = 4
A = "Sally"
# A will not overwrite a
```

### Variable Names

- Must start with a letter or underscore.
- Cannot start with a number.
- Can contain alphanumeric characters and underscores.
- Cannot be a Python keyword.

#### Legal Names
```python
myvar = "John"
my_var = "John"
_my_var = "John"
myVar = "John"
MYVAR = "John"
myvar2 = "John"
```

#### Illegal Names
```python
2myvar = "John"
my-var = "John"
my var = "John"
```

### Multi-Word Variable Names

- Use techniques like Camel Case, Pascal Case, or Snake Case for readability.

#### Examples
```python
# Camel Case
myVariableName = "John"

# Pascal Case
MyVariableName = "John"

# Snake Case
my_variable_name = "John"
```

### Assign Multiple Values

- Assign values to multiple variables in one line.

#### Example
```python
x, y, z = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)
```

### One Value to Multiple Variables

- Assign the same value to multiple variables in one line.

#### Example
```python
x = y = z = "Orange"
print(x)
print(y)
print(z)
```

### Unpack a Collection

- Extract values from a collection into variables.

#### Example
```python
fruits = ["apple", "banana", "cherry"]
x, y, z = fruits
print(x)
print(y)
print(z)
```

### Output Variables

- Use `print()` to output variables.

#### Example
```python
x = "Python is awesome"
print(x)
```

- Combine variables in `print()` using commas or `+`.

#### Example
```python
x = "Python"
y = "is"
z = "awesome"
print(x, y, z)
print(x + " " + y + " " + z)
```

### Global Variables

- Created outside functions and accessible globally.

#### Example
```python
x = "awesome"

def myfunc():
    print("Python is " + x)

myfunc()
```

- Local variables inside functions do not affect global variables.

#### Example
```python
x = "awesome"

def myfunc():
    x = "fantastic"
    print("Python is " + x)

myfunc()
print("Python is " + x)
```

### The `global` Keyword

- Use `global` to create or modify global variables inside functions.

#### Example
```python
def myfunc():
    global x
    x = "fantastic"

myfunc()
print("Python is " + x)
```

# Datatypes:

## Built-in Data Types:

In programming, data type is an important concept.

Variables can store data of different types, and different types can do different things.

Python has the following data types built-in by default, in these categories:

- **Text Type**: `str`
- **Numeric Types**: `int`, `float`, `complex`
- **Sequence Types**: `list`, `tuple`, `range`
- **Mapping Type**: `dict`
- **Set Types**: `set`, `frozenset`
- **Boolean Type**: `bool`
- **Binary Types**: `bytes`, `bytearray`, `memoryview`
- **None Type**: `NoneType`

### Getting the Data Type: type()

You can get the data type of any object by using the `type()` function:

#### Example
```python
x = 5
print(type(x))  # Output: <class 'int'>
```

### Setting the Data Type

In Python, the data type is set when you assign a value to a variable:

#### Example
```python
x = "Hello World"  # str
x = 20  # int
x = 20.5  # float
x = 1j  # complex
x = ["apple", "banana", "cherry"]  # list
x = ("apple", "banana", "cherry")  # tuple
x = range(6)  # range
x = {"name": "John", "age": 36}  # dict
x = {"apple", "banana", "cherry"}  # set
x = frozenset({"apple", "banana", "cherry"})  # frozenset
x = True  # bool
x = b"Hello"  # bytes
x = bytearray(5)  # bytearray
x = memoryview(bytes(5))  # memoryview
x = None  # NoneType
```

### Setting the Specific Data Type

If you want to specify the data type, you can use the following constructor functions:

#### Example
```python
x = str("Hello World")  # str
x = int(20)  # int
x = float(20.5)  # float
x = complex(1j)  # complex
x = list(("apple", "banana", "cherry"))  # list
x = tuple(("apple", "banana", "cherry"))  # tuple
x = range(6)  # range
x = dict(name="John", age=36)  # dict
x = set(("apple", "banana", "cherry"))  # set
x = frozenset(("apple", "banana", "cherry"))  # frozenset
x = bool(5)  # bool
x = bytes(5)  # bytes
x = bytearray(5)  # bytearray
x = memoryview(bytes(5))  # memoryview
```

### Type Casting

Type casting is the process of converting one data type to another. Python supports several built-in functions for type casting.

#### Example
```python
# Convert to integer
x = int(1)  # x will be 1
y = int(2.8)  # y will be 2
z = int("3")  # z will be 3

# Convert to float
a = float(1)  # a will be 1.0
b = float(2.8)  # b will be 2.8
c = float("3")  # c will be 3.0
d = float("4.2")  # d will be 4.2

# Convert to string
e = str(1)  # e will be '1'
f = str(2.8)  # f will be '2.8'
g = str(3)  # g will be '3'

# Convert to list
h = list((1, 2, 3))  # h will be [1, 2, 3]

# Convert to tuple
i = tuple([1, 2, 3])  # i will be (1, 2, 3)

# Convert to set
j = set([1, 2, 3])  # j will be {1, 2, 3}

# Convert to dictionary
k = dict([(1, 'one'), (2, 'two')])  # k will be {1: 'one', 2: 'two'}

# Convert to boolean
l = bool(1)  # l will be True
m = bool(0)  # m will be False
```

### Summary of Type Casting Functions

- `int(x)`: Converts `x` to an integer.
- `float(x)`: Converts `x` to a float.
- `str(x)`: Converts `x` to a string.
- `list(x)`: Converts `x` to a list.
- `tuple(x)`: Converts `x` to a tuple.
- `set(x)`: Converts `x` to a set.
- `dict(x)`: Converts `x` to a dictionary.
- `bool(x)`: Converts `x` to a boolean.

## String:

### String Methods

Note: All string methods return new values. They do not change the original string.

### Method Descriptions


- `str.capitalize()` - Converts the first character to upper case.
- `str.casefold()` - Converts string into lower case.
- `str.center(width, fillchar=' ')` - Returns a centered string. `width` specifies the total length of the string, and `fillchar` is the character to fill the padding (default is space).
- `str.count(sub, start=0, end=len(string))` - Returns the number of times a specified value occurs in a string. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.encode(encoding='utf-8', errors='strict')` - Returns an encoded version of the string. `encoding` specifies the encoding to use, and `errors` specifies the error handling scheme.
- `str.endswith(suffix, start=0, end=len(string))` - Returns true if the string ends with the specified value. `suffix` is the substring to check, `start` and `end` specify the range to check within.
- `str.expandtabs(tabsize=8)` - Sets the tab size of the string. `tabsize` specifies the number of spaces to use per tab.
- `str.find(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.format(*args, **kwargs)` - Formats specified values in a string. `args` and `kwargs` are the values to format into the string.
- `str.format_map(mapping)` - Formats specified values in a string. `mapping` is a dictionary-like object with the values to format into the string.
- `str.index(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.isalnum()` - Returns True if all characters in the string are alphanumeric.
- `str.isalpha()` - Returns True if all characters in the string are in the alphabet.
- `str.isascii()` - Returns True if all characters in the string are ASCII characters.
- `str.isdecimal()` - Returns True if all characters in the string are decimals.
- `str.isdigit()` - Returns True if all characters in the string are digits.
- `str.isidentifier()` - Returns True if the string is a valid identifier.
- `str.islower()` - Returns True if all characters in the string are lower case.
- `str.isnumeric()` - Returns True if all characters in the string are numeric.
- `str.isprintable()` - Returns True if all characters in the string are printable.
- `str.isspace()` - Returns True if all characters in the string are whitespaces.
- `str.istitle()` - Returns True if the string follows the rules of a title.
- `str.isupper()` - Returns True if all characters in the string are upper case.
- `str.join(iterable)` - Joins the elements of an iterable to the end of the string. `iterable` is the collection of elements to join.
- `str.ljust(width, fillchar=' ')` - Returns a left justified version of the string. `width` specifies the total length of the string, and `fillchar` is the character to fill the padding (default is space).
- `str.lower()` - Converts a string into lower case.
- `str.lstrip(chars=None)` - Returns a left trim version of the string. `chars` specifies the set of characters to remove (default is whitespace).
- `str.maketrans(x, y=None, z=None)` - Returns a translation table to be used in translations. `x` and `y` are strings of equal length, and `z` is a string with characters to delete.
- `str.partition(sep)` - Returns a tuple where the string is parted into three parts. `sep` is the separator to use.
- `str.replace(old, new, count=-1)` - Returns a string where a specified value is replaced with a specified value. `old` is the substring to replace, `new` is the replacement, and `count` specifies the number of replacements (default is all).
- `str.rfind(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the last position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.rindex(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the last position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.rjust(width, fillchar=' ')` - Returns a right justified version of the string. `width` specifies the total length of the string, and `fillchar` is the character to fill the padding (default is space).
- `str.rpartition(sep)` - Returns a tuple where the string is parted into three parts. `sep` is the separator to use.
- `str.rsplit(sep=None, maxsplit=-1)` - Splits the string at the specified separator, and returns a list. `sep` is the delimiter to split by (default is whitespace), and `maxsplit` specifies the maximum number of splits.
- `str.rstrip(chars=None)` - Returns a right trim version of the string. `chars` specifies the set of characters to remove (default is whitespace).
- `str.split(sep=None, maxsplit=-1)` - Splits the string at the specified separator, and returns a list. `sep` is the delimiter to split by (default is whitespace), and `maxsplit` specifies the maximum number of splits.
- `str.splitlines(keepends=False)` - Splits the string at line breaks and returns a list. `keepends` specifies whether to keep the line breaks (default is False).
- `str.startswith(prefix, start=0, end=len(string))` - Returns true if the string starts with the specified value. `prefix` is the substring to check, `start` and `end` specify the range to check within.
- `str.strip(chars=None)` - Returns a trimmed version of the string. `chars` specifies the set of characters to remove (default is whitespace).
- `str.swapcase()` - Swaps cases, lower case becomes upper case and vice versa.
- `str.title()` - Converts the first character of each word to upper case.
- `str.translate(table)` - Returns a translated string. `table` is a translation table to use.
- `str.upper()` - Converts a string into upper case.
- `str.zfill(width)` - Fills the string with a specified number of 0 values at the beginning. `width` specifies the total length of the string.

## List:

### List Methods

- `list.append(element)` - Adds an element at the end of the list. `element` is the item to be added.
- `list.clear()` - Removes all the elements from the list.
- `list.copy()` - Returns a copy of the list.
- `list.count(value)` - Returns the number of elements with the specified value. `value` is the item to be counted.
- `list.extend(iterable)` - Adds the elements of a list (or any iterable) to the end of the current list. `iterable` is the collection of elements to be added.
- `list.index(value, start=0, end=len(list))` - Returns the index of the first element with the specified value. `value` is the item to search for, `start` and `end` specify the range to search within.
- `list.insert(index, element)` - Adds an element at the specified position. `index` is the position to insert the element, and `element` is the item to be added.
- `list.pop(index=-1)` - Removes the element at the specified position. `index` is the position of the element to be removed (default is the last item).
- `list.remove(value)` - Removes the item with the specified value. `value` is the item to be removed.
- `list.reverse()` - Reverses the order of the list.
- `list.sort(key=None, reverse=False)` - Sorts the list. `key` is a function that serves as a key for the sort comparison, and `reverse` is a boolean value to sort in descending order.

## Tuples

### Tuple Methods

- `tuple.count(value)` - Returns the number of times a specified value occurs in a tuple. `value` is the element to count.
- `tuple.index(value, start=0, end=len(tuple))` - Searches the tuple for a specified value and returns the position of where it was found. `value` is the element to search for, `start` and `end` specify the range to search within.

## Sets:

### Set Methods

- `set.add(elem)` - Adds an element `elem` to the set.
- `set.clear()` - Removes all the elements from the set.
- `set.copy()` - Returns a copy of the set.
- `set.difference(*others)` - Returns a set containing the difference between this set and `others`.
- `set.difference_update(*others)` - Removes the items in this set that are also included in `others`.
- `set.discard(elem)` - Removes the specified element `elem` from the set if it is present.
- `set.intersection(*others)` - Returns a set that is the intersection of this set and `others`.
- `set.intersection_update(*others)` - Removes the items in this set that are not present in `others`.
- `set.isdisjoint(other)` - Returns whether two sets have an intersection or not.
- `set.issubset(other)` - Returns whether this set is a subset of `other`.
- `set.issuperset(other)` - Returns whether this set is a superset of `other`.
- `set.pop()` - Removes and returns an arbitrary element from the set.
- `set.remove(elem)` - Removes the specified element `elem` from the set. Raises a KeyError if `elem` is not found.
- `set.symmetric_difference(other)` - Returns a set with the symmetric differences of this set and `other`.
- `set.symmetric_difference_update(other)` - Updates this set with the symmetric differences of this set and `other`.
- `set.union(*others)` - Returns a set containing the union of this set and `others`.
- `set.update(*others)` - Updates this set with the union of this set and `others`.

### Method Shortcuts

- `set.difference(*others)` - `-`
- `set.difference_update(*others)` - `-=`
- `set.intersection(*others)` - `&`
- `set.intersection_update(*others)` - `&=`
- `set.issubset(other)` - `<=`
- `set.issuperset(other)` - `>=`
- `set.symmetric_difference(other)` - `^`
- `set.symmetric_difference_update(other)` - `^=`
- `set.union(*others)` - `|`
- `set.update(*others)` - `|=`

## Dictionary:

### Dictionary Methods

- `dict.clear()` - Removes all the elements from the dictionary.
- `dict.copy()` - Returns a copy of the dictionary.
- `dict.fromkeys(keys, value=None)` - Returns a dictionary with the specified keys and value. `keys` is an iterable of keys, and `value` is the value to set for all keys (default is None).
- `dict.get(key, default=None)` - Returns the value of the specified key. `key` is the key to look up, and `default` is the value to return if the key is not found (default is None).
- `dict.items()` - Returns a view object that displays a list of a dictionary's key-value tuple pairs.
- `dict.keys()` - Returns a view object that displays a list of all the keys in the dictionary.
- `dict.pop(key, default=None)` - Removes the element with the specified key. `key` is the key to remove, and `default` is the value to return if the key is not found (default is None).
- `dict.popitem()` - Removes the last inserted key-value pair.
- `dict.setdefault(key, default=None)` - Returns the value of the specified key. If the key does not exist, inserts the key with the specified value. `key` is the key to look up, and `default` is the value to set if the key is not found (default is None).
- `dict.update([other])` - Updates the dictionary with the specified key-value pairs. `other` can be another dictionary or an iterable of key-value pairs.
- `dict.values()` - Returns a view object that displays a list of all the values in the dictionary.

# Control Flow

## Conditional Statements

1. **if Statement**
   ```python
   if condition:
       # Code to execute if condition is True
   ```

2. **if-else Statement**
   ```python
   if condition:
       # Code to execute if condition is True
   else:
       # Code to execute if condition is False
   ```

3. **if-elif-else Statement**
   ```python
   if condition1:
       # Code to execute if condition1 is True
   elif condition2:
       # Code to execute if condition2 is True
   else:
       # Code to execute if both conditions are False
   ```

## Looping Statements

1. **for Loop**
   ```python
   for variable in iterable:
       # Code to execute for each item in iterable
   else:
       # (optional) Code to execute when the loop completes normally (no break)
   ```

2. **while Loop**
   ```python
   while condition:
       # Code to execute as long as condition is True
   else:
       # (optional) Code to execute when the loop completes normally (no break)
   ```

## Control Flow with Break and Continue

1. **break Statement**
   ```python
   for variable in iterable:
       if condition:
           break  # Exit the loop
       # Code to execute if condition is False
   ```

2. **continue Statement**
   ```python
   for variable in iterable:
       if condition:
           continue  # Skip the rest of the loop body
       # Code to execute if condition is False
   ```

## Exception and Exception Handling

### Exception Handling

- When an error occurs, Python will normally stop and generate an error message.
- These exceptions can be handled using the `try` statement.

- `try`: The block of code to be tested for exceptions.
- `except ExceptionClass_Name as e`: The block of code to be executed if a specific exception occurs. The exception instance is referred to as `e`.
- `except`: The block of code to be executed if any exception occurs.
- `else`: The block of code to be executed if no exceptions occur.
- `finally`: The block of code to be executed regardless of whether an exception occurs or not.

#### Syntax

```python
try:
    # code that may raise an exception
    pass

except ExceptionClass_Name as e:
    # code to be executed in case of specific type of exception
    # refer to exception using 'e'
    pass

except:
    # code that runs if an exception occurs
    pass

else:
    # code to be executed if no exception occurs
    pass

finally:
    # code that runs no matter what (exception or not)
    pass

```

### Raise Exceptions in Python

- To throw (raise) an exception in Python, you use the `raise` statement followed by an instance of the exception class.

#### Basic Syntax

```python
raise ExceptionType("Error message")
```

### Custom Exception

#### Syntax

```python
class CustomError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(message)

    def __str__(self):
        return f"[Error {self.code}]: {self.message}"

# Usage
try:
    raise CustomError("Something went wrong", 404)
except CustomError as e:
    print("args:", e.args)          # Output: args: ('Something went wrong',)
    print("str:", str(e))           # Output: str: [Error 404]: Something went wrong
    print("repr:", repr(e))         # Output: repr: CustomError('Something went wrong')
    print("message:", e.message)    # Output: message: Something went wrong
    print("code:", e.code)          # Output: code: 404

``` 

### Common Built-in Exceptions

| Exception Class       | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `Exception`           | Base class for all exceptions.                                              |
| `AttributeError`      | Raised when an attribute reference or assignment fails.                     |
| `EOFError`            | Raised when the `input()` function hits an end-of-file condition.           |
| `ImportError`         | Raised when an import statement fails to find the module definition.        |
| `IndexError`          | Raised when a sequence subscript is out of range.                           |
| `KeyError`            | Raised when a dictionary key is not found.                                  |
| `KeyboardInterrupt`   | Raised when the user hits the interrupt key (usually `Ctrl+C` or `Delete`). |
| `MemoryError`         | Raised when an operation runs out of memory.                                |
| `NameError`           | Raised when a local or global name is not found.                            |
| `OSError`             | Raised when a system-related operation causes an error.                     |
| `OverflowError`       | Raised when the result of an arithmetic operation is too large to be expressed. |
| `RuntimeError`        | Raised when an error is detected that doesn't fall in any of the other categories. |
| `StopIteration`       | Raised by the `next()` function to indicate that there are no further items produced by the iterator. |
| `SyntaxError`         | Raised when the parser encounters a syntax error.                           |
| `IndentationError`    | Raised when there is incorrect indentation.                                 |
| `TabError`            | Raised when indentation consists of inconsistent tabs and spaces.           |
| `SystemError`         | Raised when the interpreter finds an internal error.                        |
| `TypeError`           | Raised when an operation or function is applied to an object of inappropriate type. |
| `UnboundLocalError`   | Raised when a local variable is referenced before it has been assigned.     |
| `ValueError`          | Raised when a function receives an argument of the correct type but inappropriate value. |
| `ZeroDivisionError`   | Raised when the second argument of a division or modulo operation is zero.  |
