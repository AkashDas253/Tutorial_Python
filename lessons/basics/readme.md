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

### Assignment:

### Operations:


# Control Flow:

## Inside code:

### Loops:

### Conditional Branching:

#### for loop:

#### while loop:

### Escape Sequence

#### if statement:

#### else-if statement:

### Exception Handling:

#### Try-catch statement:

## Python Try Except

### Overview

- **try block**: Test a block of code for errors.
- **except block**: Handle the error.
- **else block**: Execute code when there is no error.
- **finally block**: Execute code regardless of the result of the try- and except blocks.

### Exception Handling

- When an error occurs, Python will normally stop and generate an error message.
- These exceptions can be handled using the `try` statement.

#### Syntax
```python
try:
    # code that may raise an exception
except:
    # code that runs if an exception occurs
```

### Example

- The `try` block will generate an exception because `x` is not defined:

#### Syntax
```python
try:
    print(x)
except:
    print("An exception occurred")
```

- Without the `try` block, the program will crash and raise an error:

#### Syntax
```python
print(x)
```

### Many Exceptions

- Define multiple exception blocks for different error types:

#### Syntax
```python
try:
    print(x)
except NameError:
    print("Variable x is not defined")
except:
    print("Something else went wrong")
```

### Else

- Use the `else` keyword to define a block of code to be executed if no errors were raised:

#### Syntax
```python
try:
    print("Hello")
except:
    print("Something went wrong")
else:
    print("Nothing went wrong")
```

### Finally

- The `finally` block will be executed regardless if the `try` block raises an error or not:

#### Syntax
```python
try:
    print(x)
except:
    print("Something went wrong")
finally:
    print("The 'try except' is finished")
```

- Useful to close objects and clean up resources:

#### Syntax
```python
try:
    f = open("demofile.txt")
    try:
        f.write("Lorum Ipsum")
    except:
        print("Something went wrong when writing to the file")
    finally:
        f.close()
except:
    print("Something went wrong when opening the file")
```

### Raise an Exception

- Use the `raise` keyword to throw an exception if a condition occurs.

#### Syntax
```python
x = -1

if x < 0:
    raise Exception("Sorry, no numbers below zero")
```

- Define what kind of error to raise and the text to print to the user:

#### Syntax
```python
x = "hello"

if not type(x) is int:
    raise TypeError("Only integers are allowed")
```

# OOPs

## Python Classes and Objects

### Python Classes/Objects

- Python is an object-oriented programming language.
- Almost everything in Python is an object, with its properties and methods.
- A Class is like an object constructor, or a "blueprint" for creating objects.

### Create a Class

- To create a class, use the keyword `class`:

#### Syntax
```python
class ClassName:
    property_name = value
```

### Create Object

- Use the class to create objects:

#### Syntax
```python
object_name = ClassName()
print(object_name.property_name)
```

### The `__init__()` Function

- The `__init__()` function is executed when the class is being initiated.
- Use it to assign values to object properties or perform other operations when the object is created.

#### Syntax
```python
class ClassName:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

object_name = ClassName(value1, value2)
print(object_name.property1)
print(object_name.property2)
```

### The `__str__()` Function

- The `__str__()` function controls the string representation of the class object.

#### Syntax
```python
class ClassName:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

    def __str__(self):
        return f"{self.property1}({self.property2})"

object_name = ClassName(value1, value2)
print(object_name)
```

### Object Methods

- Methods in objects are functions that belong to the object.

#### Syntax
```python
class ClassName:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

    def method_name(self):
        print("Message " + self.property1)

object_name = ClassName(value1, value2)
object_name.method_name()
```

### The `self` Parameter

- The `self` parameter is a reference to the current instance of the class.
- It is used to access variables that belong to the class.
- It can be named anything, but it must be the first parameter of any function in the class.

#### Syntax
```python
class ClassName:
    def __init__(custom_self, param1, param2):
        custom_self.property1 = param1
        custom_self.property2 = param2

    def method_name(custom_self):
        print("Message " + custom_self.property1)

object_name = ClassName(value1, value2)
object_name.method_name()
```

## Python Inheritance

### Inheritance Overview

- Inheritance allows defining a class that inherits all methods and properties from another class.
- **Parent class**: The class being inherited from (base class).
- **Child class**: The class that inherits from another class (derived class).

### Create a Parent Class

- Any class can be a parent class. The syntax is the same as creating any other class.

#### Syntax
```python
class ParentClass:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

    def method_name(self):
        print(self.property1, self.property2)

# Create an object and execute a method
obj = ParentClass(value1, value2)
obj.method_name()
```

### Create a Child Class

- To create a class that inherits from another class, pass the parent class as a parameter when creating the child class.

#### Syntax
```python
class ChildClass(ParentClass):
    pass

# Create an object and execute an inherited method
obj = ChildClass(value1, value2)
obj.method_name()
```

### Add the `__init__()` Function

- Adding the `__init__()` function to the child class overrides the parent's `__init__()` function.
- To keep the inheritance of the parent's `__init__()` function, call the parent's `__init__()` function within the child's `__init__()`.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2):
        ParentClass.__init__(self, param1, param2)
```

### Use the `super()` Function

- The `super()` function allows the child class to inherit all methods and properties from its parent without explicitly naming the parent class.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2):
        super().__init__(param1, param2)
```

### Add Properties

- Add properties to the child class by defining them in the `__init__()` function.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2, param3):
        super().__init__(param1, param2)
        self.property3 = param3

# Create an object with the new property
obj = ChildClass(value1, value2, value3)
```

### Add Methods

- Add methods to the child class by defining them within the class.
- If a method in the child class has the same name as a method in the parent class, it overrides the parent method.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2, param3):
        super().__init__(param1, param2)
        self.property3 = param3

    def new_method(self):
        print("Welcome", self.property1, self.property2, "to the class of", self.property3)

# Create an object and execute the new method
obj = ChildClass(value1, value2, value3)
obj.new_method()
```

## Python Iterators

### Python Iterators

- An iterator is an object that contains a countable number of values.
- It can be iterated upon, meaning you can traverse through all the values.
- An iterator implements the iterator protocol, which consists of the methods `__iter__()` and `__next__()`.

### Iterator vs Iterable

- Lists, tuples, dictionaries, and sets are iterable objects.
- These objects have an `iter()` method to get an iterator.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))
```

- Strings are also iterable objects, containing a sequence of characters.

#### Syntax
```python
mystr = "banana"
myit = iter(mystr)

print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
```

### Looping Through an Iterator

- Use a `for` loop to iterate through an iterable object.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)
```

#### Syntax
```python
mystr = "banana"

for x in mystr:
  print(x)
```

- The `for` loop creates an iterator object and executes the `next()` method for each loop.

### Create an Iterator

- Implement the methods `__iter__()` and `__next__()` to create an iterator.

#### Syntax
```python
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    x = self.a
    self.a += 1
    return x

myclass = MyNumbers()
myiter = iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
```

### StopIteration

- Use the `StopIteration` statement to prevent the iteration from going on forever.

#### Syntax
```python
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    if self.a <= 20:
      x = self.a
      self.a += 1
      return x
    else:
      raise StopIteration

myclass = MyNumbers()
myiter = iter(myclass)

for x in myiter:
  print(x)
```

## Python Polymorphism

### Definition

- Polymorphism means "many forms".
- In programming, it refers to methods/functions/operators with the same name that can be executed on many objects or classes.

### Function Polymorphism

- Example of a Python function that can be used on different objects: `len()` function.

#### String

- For strings, `len()` returns the number of characters.

#### Syntax
```python
x = "Hello World```"
print(len(x))
```

#### Tuple

- For tuples, `len()` returns the number of items in the tuple.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")
print(len(mytuple))
```

#### Dictionary

- For dictionaries, `len()` returns the number of key/value pairs in the dictionary.

#### Syntax
```python
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(len(thisdict))
```

### Class Polymorphism

- Polymorphism is often used in class methods, where multiple classes have the same method name.

#### Example

- Different classes with the same method:

#### Syntax
```python
class Car:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Drive```")

class Boat:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Sail```")

class Plane:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Fly```")

car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Touring 20")
plane1 = Plane("Boeing", "747")

for x in (car1, boat1, plane1):
  x.move()
```

- The `for` loop at the end demonstrates polymorphism by executing the same method for all three classes.

### Inheritance Class Polymorphism

- Child classes with the same name can use polymorphism.

#### Example

- Create a parent class `Vehicle` and make `Car`, `Boat`, `Plane` child classes of `Vehicle`.

#### Syntax
```python
class Vehicle:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Move```")

class Car(Vehicle):
  pass

class Boat(Vehicle):
  def move(self):
    print("Sail```")

class Plane(Vehicle):
  def move(self):
    print("Fly```")

car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Touring 20")
plane1 = Plane("Boeing", "747")

for x in (car1, boat1, plane1):
  print(x.brand)
  print(x.model)
  x.move()
```

- Child classes inherit properties and methods from the parent class.
- The `Car` class is empty but inherits `brand`, `model`, and `move()` from `Vehicle`.
- The `Boat` and `Plane` classes override the `move()` method.
- Polymorphism allows executing the same method for all classes.

## Python Scope

### Scope Definition

- A variable is only available from inside the region it is created. This is called scope.

### Local Scope

- A variable created inside a function belongs to the local scope of that function and can only be used inside that function.

#### Syntax
```python
def myfunc():
    x = 300
    print(x)

myfunc()
```

### Function Inside Function

- A local variable can be accessed from a function within the function.

#### Syntax
```python
def myfunc():
    x = 300
    def myinnerfunc():
        print(x)
    myinnerfunc()

myfunc()
```

### Global Scope

- A variable created in the main body of the Python code is a global variable and belongs to the global scope.
- Global variables are available from within any scope, global and local.

#### Syntax
```python
x = 300

def myfunc():
    print(x)

myfunc()
print(x)
```

### Naming Variables

- If you operate with the same variable name inside and outside of a function, Python will treat them as two separate variables, one available in the global scope (outside the function) and one available in the local scope (inside the function).

#### Syntax
```python
x = 300

def myfunc():
    x = 200
    print(x)

myfunc()
print(x)
```

### Global Keyword

- If you need to create a global variable but are stuck in the local scope, you can use the `global` keyword.
- The `global` keyword makes the variable global.

#### Syntax
```python
def myfunc():
    global x
    x = 300

myfunc()
print(x)
```

- Use the `global` keyword if you want to make a change to a global variable inside a function.

#### Syntax
```python
x = 300

def myfunc():
    global x
    x = 200

myfunc()
print(x)
```

### Nonlocal Keyword

- The `nonlocal` keyword is used to work with variables inside nested functions.
- The `nonlocal` keyword makes the variable belong to the outer function.

#### Syntax
```python
def myfunc1():
    x = "Jane"
    def myfunc2():
        nonlocal x
        x = "hello"
    myfunc2()
    return x

print(myfunc1())
```

