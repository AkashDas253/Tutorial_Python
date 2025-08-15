## Python String Methods 

### String Creation

```python
s = "Hello"
```

### Case Conversion

* `str.lower()` — Convert to lowercase
* `str.upper()` — Convert to uppercase
* `str.capitalize()` — Capitalize first character, rest lowercase
* `str.title()` — Capitalize first character of each word
* `str.swapcase()` — Swap uppercase to lowercase and vice versa
* `str.casefold()` — Aggressive lowercase conversion (for caseless matching)

### Search & Find

* `str.find(sub, start=0, end=len(str))` — Return index of first occurrence or `-1`
* `str.rfind(sub, start=0, end=len(str))` — Last occurrence
* `str.index(sub, start=0, end=len(str))` — Like `find()` but raises `ValueError` if not found
* `str.rindex(sub, start=0, end=len(str))` — Last occurrence, raises if not found
* `str.count(sub, start=0, end=len(str))` — Count occurrences

### Check Content (Boolean Methods)

* `str.startswith(prefix, start=0, end=len(str))`
* `str.endswith(suffix, start=0, end=len(str))`
* `str.isalpha()` — All alphabetic
* `str.isdigit()` — All digits
* `str.isdecimal()` — Decimal characters only
* `str.isnumeric()` — Numeric characters
* `str.isalnum()` — Alphanumeric
* `str.isascii()` — ASCII only
* `str.isspace()` — Whitespace only
* `str.islower()` — Lowercase only
* `str.isupper()` — Uppercase only
* `str.istitle()` — Title case only

### Replace & Modify

* `str.replace(old, new, count=-1)` — Replace occurrences
* `str.strip(chars=None)` — Remove leading & trailing chars (default: whitespace)
* `str.lstrip(chars=None)` — Remove leading chars
* `str.rstrip(chars=None)` — Remove trailing chars
* `str.removeprefix(prefix)` — Remove prefix if present
* `str.removesuffix(suffix)` — Remove suffix if present

### Split & Join

* `str.split(sep=None, maxsplit=-1)` — Split into list
* `str.rsplit(sep=None, maxsplit=-1)` — Split from right
* `str.splitlines(keepends=False)` — Split by newline
* `sep.join(iterable)` — Join iterable with separator

### Alignment & Padding

* `str.center(width, fillchar=' ')`
* `str.ljust(width, fillchar=' ')`
* `str.rjust(width, fillchar=' ')`
* `str.zfill(width)` — Pad with zeros on left

### Encoding & Translation

* `str.encode(encoding='utf-8', errors='strict')` — Encode to bytes
* `str.maketrans(x, y=None, z=None)` — Create translation table
* `str.translate(table)` — Apply translation table

### Formatting

* `str.format(*args, **kwargs)`
* `str.format_map(mapping)`
* `f"{variable}"` — f-strings
* `%` formatting — `"%s" % value`

### Examples

```python
# Case methods
"hello".upper()        # 'HELLO'
"HELLO".lower()        # 'hello'
"hello world".title()  # 'Hello World'

# Search
"hello".find("l")      # 2
"hello".count("l")     # 2
"hello".startswith("he")  # True

# Replace
"banana".replace("a", "o")  # 'bonono'

# Strip
"   hi  ".strip()      # 'hi'

# Split & Join
"one,two,three".split(",")   # ['one', 'two', 'three']
"-".join(["a", "b", "c"])    # 'a-b-c'

# Alignment
"hi".center(6, "-")    # '--hi--'

# Encoding
"café".encode()        # b'caf\xc3\xa9'

# Formatting
"Hello {}".format("World")   # 'Hello World'
name = "Alice"
f"Hi {name}"           # 'Hi Alice'
```

---
