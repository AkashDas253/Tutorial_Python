## `re` Module – Regular Expressions

### Purpose

* Provides functions for searching, matching, and manipulating strings using **regular expressions**.
* Supports **Perl-style regex patterns**.

---

## Key Functions

| Function         | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `re.match()`     | Matches a pattern at the beginning of a string.                     |
| `re.fullmatch()` | Matches the entire string to the pattern.                           |
| `re.search()`    | Searches the first occurrence of a pattern in the string.           |
| `re.findall()`   | Returns all non-overlapping matches as a list.                      |
| `re.finditer()`  | Returns an iterator yielding match objects for all matches.         |
| `re.sub()`       | Replaces matches with a replacement string.                         |
| `re.subn()`      | Same as `sub()` but returns a tuple `(new_string, number_of_subs)`. |
| `re.split()`     | Splits a string by the matches of the pattern.                      |
| `re.compile()`   | Compiles a regex pattern for reuse.                                 |
| `re.escape()`    | Escapes all special characters in a string.                         |

---

## Flags

| Flag                     | Meaning                                             |
| ------------------------ | --------------------------------------------------- |
| `re.IGNORECASE` (`re.I`) | Case-insensitive matching.                          |
| `re.MULTILINE` (`re.M`)  | `^` and `$` match start/end of each line.           |
| `re.DOTALL` (`re.S`)     | `.` matches newline characters as well.             |
| `re.VERBOSE` (`re.X`)    | Allows whitespace and comments in patterns.         |
| `re.ASCII` (`re.A`)      | Makes `\w`, `\b`, `\s` match only ASCII characters. |
| `re.UNICODE`             | Enables Unicode matching (default in Python 3).     |

---

## Common Pattern Elements

| Pattern         | Meaning                                                 |                   |
| --------------- | ------------------------------------------------------- | ----------------- |
| `.`             | Any character except newline (unless `DOTALL` is used). |                   |
| `^`             | Start of string/line.                                   |                   |
| `$`             | End of string/line.                                     |                   |
| `*`             | 0 or more repetitions.                                  |                   |
| `+`             | 1 or more repetitions.                                  |                   |
| `?`             | 0 or 1 repetition.                                      |                   |
| `{m}`           | Exactly m repetitions.                                  |                   |
| `{m,n}`         | Between m and n repetitions.                            |                   |
| `[abc]`         | Matches `a`, `b`, or `c`.                               |                   |
| `[^abc]`        | Matches any character except `a`, `b`, `c`.             |                   |
| `\d`            | Digit.                                                  |                   |
| `\D`            | Non-digit.                                              |                   |
| `\w`            | Word character.                                         |                   |
| `\W`            | Non-word character.                                     |                   |
| `\s`            | Whitespace.                                             |                   |
| `\S`            | Non-whitespace.                                         |                   |
| `()`            | Capturing group.                                        |                   |
| `(?:...)`       | Non-capturing group.                                    |                   |
| `(?P<name>...)` | Named capturing group.                                  |                   |
| \`              | \`                                                      | Alternation (OR). |

---

## Syntax Examples

### Match at Start

```python
import re

result = re.match(r"\d+", "123abc")  # Matches digits at start
if result:
    print(result.group())  # Output: 123
```

### Full Match

```python
result = re.fullmatch(r"\d+", "123")  # Entire string must be digits
print(result.group() if result else "No match")  # Output: 123
```

### Search Anywhere

```python
result = re.search(r"abc", "123abc456")
print(result.group())  # Output: abc
```

### Find All Matches

```python
matches = re.findall(r"\d+", "abc123def456")
print(matches)  # Output: ['123', '456']
```

### Find Iteratively

```python
for match in re.finditer(r"\d+", "abc123def456"):
    print(match.group())  # Output: 123 then 456
```

### Replace Matches

```python
result = re.sub(r"\d+", "#", "abc123def456")
print(result)  # Output: abc#def#
```

### Replace with Count

```python
result, count = re.subn(r"\d+", "#", "abc123def456")
print(result, count)  # Output: abc#def# 2
```

### Split String

```python
parts = re.split(r"\d+", "abc123def456ghi")
print(parts)  # Output: ['abc', 'def', 'ghi']
```

### Compile for Reuse

```python
pattern = re.compile(r"\d+")
print(pattern.findall("abc123def456"))  # Output: ['123', '456']
```

### Escape Special Characters

```python
escaped = re.escape("a.b*c?")
print(escaped)  # Output: a\.b\*c\?
```

---
