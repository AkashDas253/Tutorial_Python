## Python String Operations

### String Creation

* Single quotes `'...'`
* Double quotes `"..."`
* Triple quotes `'''...'''` / `"""..."""` for multi-line
* `str()` constructor

### String Access & Slicing

* Indexing: `s[i]` (0-based, negative indexing allowed)
* Slicing: `s[start:end:step]` (end exclusive)
* Reverse: `s[::-1]`

### String Concatenation & Repetition

* Concatenation: `s1 + s2`
* Repetition: `s * n`

### String Formatting

* Old style: `"Hello %s" % name`
* `str.format()`: `"Hello {}".format(name)`
* f-strings: `f"Hello {name}"`
* Template strings (from `string` module)

### String Methods — Transformation

* Case conversion: `upper()`, `lower()`, `title()`, `capitalize()`, `swapcase()`
* Alignment: `center(width, char)`, `ljust(width, char)`, `rjust(width, char)`
* Padding: `zfill(width)`
* Strip: `strip()`, `lstrip()`, `rstrip()`
* Replace: `replace(old, new, count)`

### String Methods — Search & Check

* Search: `find(sub)`, `rfind(sub)`, `index(sub)`, `rindex(sub)`
* Membership: `in`, `not in`
* Count occurrences: `count(sub)`
* Starts/ends: `startswith(prefix)`, `endswith(suffix)`

### String Methods — Testing (Return Bool)

* `isalnum()`, `isalpha()`, `isdigit()`, `isdecimal()`
* `isnumeric()`, `isidentifier()`
* `islower()`, `isupper()`, `istitle()`
* `isspace()`

### String Splitting & Joining

* `split(sep, maxsplit)`, `rsplit(sep, maxsplit)`
* `splitlines(keepends)`
* Join iterable: `sep.join(iterable)`
* Partition: `partition(sep)`, `rpartition(sep)`

### String Encoding & Decoding

* Encode: `encode(encoding, errors)`
* Decode (bytes → str): `decode(encoding, errors)`

### Escape Sequences

* `\n` newline, `\t` tab, `\\` backslash, `\'`, `\"`, `\r`, `\b`
* Raw string: `r"..."`

### String Constants (`string` module)

* `string.ascii_letters`, `ascii_lowercase`, `ascii_uppercase`
* `digits`, `punctuation`, `whitespace`, `printable`

---

### Example Usage

```python
# Creation
s = "Hello World"

# Access & Slicing
print(s[0])       # 'H'
print(s[-1])      # 'd'
print(s[0:5])     # 'Hello'
print(s[::-1])    # 'dlroW olleH'

# Transformation
print(s.upper())  # 'HELLO WORLD'
print(s.title())  # 'Hello World'
print(s.replace("World", "Python"))  # 'Hello Python'

# Search & Check
print(s.find("World"))   # 6
print(s.startswith("He")) # True
print("Hello" in s)      # True

# Split & Join
words = s.split()
print(words)             # ['Hello', 'World']
print("-".join(words))   # 'Hello-World'

# Encoding
encoded = s.encode("utf-8")
print(encoded)           # b'Hello World'
print(encoded.decode("utf-8"))  # 'Hello World'
```

---
