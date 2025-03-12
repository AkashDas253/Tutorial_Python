## Python Syntax Rules  

### Indentation  
- Indentation is mandatory and replaces braces `{}` used in other languages.  
- Default indentation level: 4 spaces (not tabs).  
- Blocks must be consistently indented.  

### Case Sensitivity  
- Python is case-sensitive: `variable`, `Variable`, and `VARIABLE` are different.  

### Statements  
- Each statement is usually on a new line.  
- Multiple statements can be written on one line using `;`, but it's discouraged.  
- Line continuation:  
  - Implicit using parentheses `()`, brackets `[]`, or braces `{}`.  
  - Explicit using `\` at the end of the line.  

### Comments  
- Single-line: `# This is a comment`  
- Multi-line: Use triple quotes `"""` or `'''`, typically for docstrings.  

### Variables  
- No explicit declaration required.  
- Dynamic typing allows reassignment to different types.  

### Naming Rules  
- Must start with a letter (A-Z, a-z) or underscore `_`.  
- Followed by letters, digits (0-9), or underscores.  
- Cannot be a Python keyword.  
- Convention:  
  - Variables/functions: `lower_case_with_underscores`  
  - Constants: `ALL_CAPS`  
  - Classes: `PascalCase`  

### Operators  
- Common arithmetic: `+`, `-`, `*`, `/`, `%`, `//`, `**`  
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`  
- Logical: `and`, `or`, `not`  
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`  

### Functions  
- Defined using `def`:  
  ```python
  def function_name(parameters):
      return value
  ```  
- Default arguments, keyword arguments, and variable-length arguments supported.  

### Loops  
- `for` loops iterate over sequences.  
- `while` loops execute while a condition is `True`.  
- `break` exits the loop, `continue` skips an iteration, `pass` does nothing.  

### Conditionals  
- `if`, `elif`, and `else` used for branching.  
- No `switch` statement in Python.  

### Exception Handling  
- Use `try`, `except`, `else`, `finally`.  
  ```python
  try:
      risky_code()
  except ExceptionType as e:
      handle_exception()
  else:
      execute_if_no_exception()
  finally:
      always_execute()
  ```  

### Importing Modules  
- `import module_name`  
- `from module_name import function_name`  
- `import module_name as alias`  

### File Handling  
- Use `with open(filename, mode) as file:` for automatic file closing.  

### Classes & Objects  
- Defined using `class ClassName:`  
- `self` represents instance variables and methods.  
- Inheritance and method overriding supported.  

### List Comprehension  
- Concise way to create lists:  
  ```python
  squares = [x**2 for x in range(10)]
  ```  

### Lambda Functions  
- Anonymous functions:  
  ```python
  add = lambda x, y: x + y
  ```  

### Docstrings  
- First statement inside a function/class/module should be a string with `"""Triple quotes"""`.  

### Type Hinting (Optional)  
- Use `->` for return types and `:` for parameter types:  
  ```python
  def add(x: int, y: int) -> int:
      return x + y
  ```