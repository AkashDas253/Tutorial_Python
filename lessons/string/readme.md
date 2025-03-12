## **String (`str`) in Python**  

### **Definition**  
- A sequence of characters enclosed in single (`'`), double (`"`) or triple (`'''` or `"""`) quotes.  
- Strings are **immutable** (cannot be changed after creation).  

```python
s1 = 'Hello'
s2 = "World"
s3 = '''Multiline
string'''
```

---

### **String Indexing & Slicing**  
| Operation | Example | Result |
|-----------|---------|--------|
| Indexing | `"Python"[0]` | `'P'` |
| Negative Indexing | `"Python"[-1]` | `'n'` |
| Slicing | `"Python"[1:4]` | `'yth'` |
| Step Slicing | `"Python"[::2]` | `'Pto'` |
| Reverse String | `"Python"[::-1]` | `'nohtyP'` |

```python
s = "Python"
print(s[0])  # P
print(s[-1])  # n
print(s[1:4])  # yth
print(s[::-1])  # nohtyP
```

---

### **String Operations**  
| Operator | Example | Result |
|----------|---------|--------|
| Concatenation (`+`) | `"Hello " + "World"` | `"Hello World"` |
| Repetition (`*`) | `"Hi" * 3` | `"HiHiHi"` |
| Membership (`in`) | `'a' in "apple"` | `True` |
| Length (`len()`) | `len("Python")` | `6` |

```python
print("Hello " + "World")  # Hello World
print("Hi" * 3)  # HiHiHi
print('a' in "apple")  # True
print(len("Python"))  # 6
```

---

### **String Methods**
| Method | Description | Example | Result |
|--------|------------|---------|--------|
| `s.lower()` | Converts to lowercase | `"Hello".lower()` | `"hello"` |
| `s.upper()` | Converts to uppercase | `"Hello".upper()` | `"HELLO"` |
| `s.title()` | Capitalizes each word | `"hello world".title()` | `"Hello World"` |
| `s.strip()` | Removes spaces | `" hello ".strip()` | `"hello"` |
| `s.replace(a, b)` | Replaces substring | `"apple".replace("p", "b")` | `"abble"` |
| `s.split()` | Splits into list | `"a,b,c".split(",")` | `['a', 'b', 'c']` |
| `",".join(lst)` | Joins list into string | `",".join(['a', 'b', 'c'])` | `"a,b,c"` |

```python
s = "  Hello World  "
print(s.strip())  # "Hello World"
print(s.lower())  # "  hello world  "
print(s.replace("World", "Python"))  # "  Hello Python  "
print("a,b,c".split(","))  # ['a', 'b', 'c']
print("-".join(["Python", "Java"]))  # "Python-Java"
```

---

### **String Formatting**
| Method | Example | Result |
|--------|---------|--------|
| `f""` | `f"Name: {name}"` | `"Name: John"` |
| `format()` | `"{} is {}".format("Python", "fun")` | `"Python is fun"` |
| `%` Formatting | `"%s is %d" % ("Age", 25)` | `"Age is 25"` |

```python
name = "John"
age = 25
print(f"My name is {name} and I am {age} years old.")  # My name is John and I am 25 years old
print("{} is {}".format("Python", "fun"))  # Python is fun
print("%s is %d" % ("Age", 25))  # Age is 25
```

---

### **Escape Characters**
| Character | Meaning | Example | Output |
|-----------|---------|---------|--------|
| `\n` | New line | `"Hello\nWorld"` | `"Hello"` `"World"` |
| `\t` | Tab | `"Hello\tWorld"` | `"Hello   World"` |
| `\'` | Single quote | `'It\'s a pen'` | `"It's a pen"` |
| `\"` | Double quote | `"He said \"Hello\""` | `"He said "Hello""` |
| `\\` | Backslash | `"C:\\User\\"` | `"C:\User\"` |

```python
print("Hello\nWorld")  # New line
print("Hello\tWorld")  # Tab space
print("He said \"Python is great\"")  # Escape double quote
```

---

### **Checking String Content**
| Method | Example | Result |
|--------|---------|--------|
| `s.startswith(x)` | `"Python".startswith("Py")` | `True` |
| `s.endswith(x)` | `"Python".endswith("on")` | `True` |
| `s.isalpha()` | `"Python".isalpha()` | `True` |
| `s.isdigit()` | `"123".isdigit()` | `True` |
| `s.isalnum()` | `"Python3".isalnum()` | `True` |
| `s.isspace()` | `"   ".isspace()` | `True` |

```python
s = "Python3"
print(s.isalpha())  # False (contains a digit)
print(s.isalnum())  # True (letters and numbers)
print("123".isdigit())  # True
```

---

### **Multi-line Strings**  
```python
text = """This is
a multi-line
string."""
```

---

### **Raw Strings (`r""`)**
- Ignores escape sequences.
```python
print(r"C:\new\name")  # C:\new\name (without special interpretation)
```

---
---

## String

### Properties and usage of String

#### Properties of Strings

- **Immutable**: Strings cannot be changed after creation.
- **Indexed**: Strings are indexed by position, starting from 0.
- **Iterable**: Strings can be iterated over.

#### Creating Strings

1. **Single Quotes**
   ```python
   my_str = 'Hello, World```'
   ```

2. **Double Quotes**
   ```python
   my_str = "Hello, World```"
   ```

3. **Triple Quotes** (for multi-line strings)
   ```python
   my_str = '''Hello,
   World```'''
   ```

#### Accessing Characters

1. **Using Indexing**
   ```python
   char = my_str[0]  # First character
   ```

2. **Using Slicing**
   ```python
   substring = my_str[0:5]  # Substring from index 0 to 4
   ```

#### String Operations

1. **Concatenation**
   ```python
   new_str = my_str + " How are you?"
   ```

2. **Repetition**
   ```python
   repeated_str = my_str * 3
   ```

3. **Length**
   ```python
   length = len(my_str)
   ```

4. **Membership**
   ```python
   if "Hello" in my_str:
       # Code to execute if "Hello" is in my_str
   ```

#### String Methods

1. **`upper` Method**: Converts all characters to uppercase
   ```python
   upper_str = my_str.upper()
   ```

2. **`lower` Method**: Converts all characters to lowercase
   ```python
   lower_str = my_str.lower()
   ```

3. **`strip` Method**: Removes leading and trailing whitespace
   ```python
   stripped_str = my_str.strip()
   ```

4. **`split` Method**: Splits the string into a list of substrings
   ```python
   words = my_str.split()
   ```

5. **`join` Method**: Joins a list of strings into a single string
   ```python
   joined_str = " ".join(words)
   ```

6. **`replace` Method**: Replaces occurrences of a substring with another substring
   ```python
   replaced_str = my_str.replace("World", "Python")
   ```

7. **`find` Method**: Returns the index of the first occurrence of a substring
   ```python
   index = my_str.find("World")
   ```

8. **`count` Method**: Returns the number of occurrences of a substring
   ```python
   count = my_str.count("l")
   ```

9. **`startswith` Method**: Checks if the string starts with a specified substring
   ```python
   starts = my_str.startswith("Hello")
   ```

10. **`endswith` Method**: Checks if the string ends with a specified substring
    ```python
    ends = my_str.endswith("```")
    ```

11. **`format` Method**: Formats the string using placeholders
    ```python
    formatted_str = "Hello, {}```".format("Python")
    ```

12. **`isalpha` Method**: Checks if all characters in the string are alphabetic
    ```python
    is_alpha = my_str.isalpha()
    ```

13. **`isdigit` Method**: Checks if all characters in the string are digits
    ```python
    is_digit = my_str.isdigit()
    ```

14. **`isalnum` Method**: Checks if all characters in the string are alphanumeric
    ```python
    is_alnum = my_str.isalnum()
    ```

15. **`title` Method**: Converts the first character of each word to uppercase
    ```python
    title_str = my_str.title()
    ```

16. **`capitalize` Method**: Converts the first character to uppercase and the rest to lowercase
    ```python
    capitalized_str = my_str.capitalize()
    ```

17. **`swapcase` Method**: Swaps the case of all characters
    ```python
    swapped_str = my_str.swapcase()
    ```

#### String Formatting

1. **Using `format` Method**
   ```python
   formatted_str = "Hello, {}```".format("Python")
   ```

2. **Using f-Strings (Python 3.6+)**
   ```python
   name = "Python"
   formatted_str = f"Hello, {name}```"
   ```

3. **Using `%` Operator**
   ```python
   formatted_str = "Hello, %s```" % "Python"
   ```

### String Method Descriptions


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