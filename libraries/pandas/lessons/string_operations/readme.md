
## **Text/String Operations in Pandas**

Text data manipulation in Pandas is handled through **vectorized string methods**, accessed via `.str` accessor on Series.

---

### Accessing String Methods

Use `.str` accessor to apply string methods on Series:

```python
df['col'].str.method()  # Applies string method element-wise
```

---

### **String Properties & Basic Inspection**

| Method / Property       | Description                                 | Example                               |
|-------------------------|---------------------------------------------|---------------------------------------|
| `str.len()`             | Length of each string                       | `df['name'].str.len()`                |
| `str.isalpha()`         | Alphabetic characters only?                 | `df['col'].str.isalpha()`             |
| `str.isnumeric()`       | Numeric characters only?                    | `df['col'].str.isnumeric()`           |
| `str.isdigit()`         | Digits only?                                | `df['col'].str.isdigit()`             |
| `str.isspace()`         | Whitespace only?                            | `df['col'].str.isspace()`             |
| `str.isalnum()`         | Alphanumeric?                               | `df['col'].str.isalnum()`             |
| `str.islower()`         | All characters lowercase?                   | `df['col'].str.islower()`             |
| `str.isupper()`         | All characters uppercase?                   | `df['col'].str.isupper()`             |
| `str.istitle()`         | Title-case string?                          | `df['col'].str.istitle()`             |

---

### **Case Conversion**

| Method             | Description                     | Example                        |
|--------------------|---------------------------------|--------------------------------|
| `str.lower()`      | Convert to lowercase            | `df['col'].str.lower()`        |
| `str.upper()`      | Convert to uppercase            | `df['col'].str.upper()`        |
| `str.title()`      | Convert to title case           | `df['col'].str.title()`        |
| `str.capitalize()` | Capitalize first letter         | `df['col'].str.capitalize()`   |
| `str.swapcase()`   | Swap upper and lower case       | `df['col'].str.swapcase()`     |

---

### **Searching & Matching**

| Method             | Description                                  | Example                             |
|--------------------|----------------------------------------------|-------------------------------------|
| `str.contains(pat)`| Check for presence of pattern (regex by default) | `df['col'].str.contains("abc")` |
| `str.startswith(pat)` | Starts with pattern                      | `df['col'].str.startswith("Mr")`    |
| `str.endswith(pat)`   | Ends with pattern                        | `df['col'].str.endswith(".com")`    |
| `str.match(pat)`   | Regex match                                 | `df['col'].str.match("^[A-Z]")`     |
| `str.find(sub)`    | Lowest index of substring                   | `df['col'].str.find("cat")`         |
| `str.rfind(sub)`   | Highest index of substring                  | `df['col'].str.rfind("cat")`        |

---

### **Substring Extraction**

| Method                 | Description                                 | Example                                 |
|------------------------|---------------------------------------------|-----------------------------------------|
| `str.slice(start, end)`| Extract substring by position               | `df['col'].str.slice(0, 5)`             |
| `str[0:5]`             | Shortcut using indexing                     | `df['col'].str[0:5]`                    |
| `str.get(i)`           | Get i-th character                          | `df['col'].str.get(3)`                  |
| `str.extract(pattern)` | Extract from regex pattern as new DataFrame | `df['col'].str.extract(r'(\d+)-(\w+)')` |
| `str.extractall(pattern)` | Extract all matches into rows           | `df['col'].str.extractall(r'(\d+)')`    |

---

### **Replacement & Removal**

| Method                    | Description                             | Example                                  |
|---------------------------|-----------------------------------------|------------------------------------------|
| `str.replace(pat, repl)`  | Replace string or pattern                | `df['col'].str.replace("old", "new")`    |
| `str.strip()`             | Remove leading/trailing whitespace       | `df['col'].str.strip()`                  |
| `str.lstrip()` / `rstrip()` | Left/right strip                     | `df['col'].str.lstrip("_")`              |
| `str.removeprefix(p)`     | Remove prefix (v3.9+)                    | `df['col'].str.removeprefix("http://")`  |
| `str.removesuffix(s)`     | Remove suffix (v3.9+)                    | `df['col'].str.removesuffix(".com")`     |

---

### **Splitting & Joining**

| Method                 | Description                                 | Example                                |
|------------------------|---------------------------------------------|----------------------------------------|
| `str.split(sep)`       | Split by delimiter into list                | `df['col'].str.split(",")`             |
| `str.rsplit(sep)`      | Right-side split                            | `df['col'].str.rsplit(",", n=1)`       |
| `str.partition(sep)`   | Split into 3 parts: before, sep, after      | `df['col'].str.partition("@")`         |
| `str.rpartition(sep)`  | Right partition                             | `df['col'].str.rpartition(".")`        |
| `str.cat(sep)`         | Concatenate string values                   | `df['col'].str.cat(sep=", ")`          |
| `str.join(sep)`        | Join each character with separator          | `df['col'].str.join("-")`              |

---

### **Padding & Formatting**

| Method                 | Description                                 | Example                                  |
|------------------------|---------------------------------------------|------------------------------------------|
| `str.zfill(width)`     | Pad numeric string with zeros               | `df['col'].str.zfill(5)`                 |
| `str.pad(width, side, fillchar)` | Pad strings on any side        | `df['col'].str.pad(10, side='right', fillchar='_')` |
| `str.center(width)`    | Center align string                         | `df['col'].str.center(12)`               |
| `str.ljust(width)`     | Left align                                  | `df['col'].str.ljust(12)`                |
| `str.rjust(width)`     | Right align                                 | `df['col'].str.rjust(12)`                |
| `str.format()`         | Format strings like Python `.format()`      | `df['col'].str.format()`                 |

---

### **Encoding & Decoding**

| Method                 | Description                                 | Example                                |
|------------------------|---------------------------------------------|----------------------------------------|
| `str.encode()`         | Encode strings to bytes                     | `df['col'].str.encode('utf-8')`        |
| `str.decode()`         | Decode byte strings                         | `df['col'].str.decode('utf-8')` *(if dtype is bytes)* |

---

### **Miscellaneous**

| Method                 | Description                                 | Example                                |
|------------------------|---------------------------------------------|----------------------------------------|
| `str.repeat(n)`        | Repeat string `n` times                     | `df['col'].str.repeat(2)`              |
| `str.wrap(width)`      | Wrap long strings into lines                | `df['col'].str.wrap(20)`               |
| `str.normalize(form)`  | Unicode normalization                       | `df['col'].str.normalize('NFC')`       |

---
