# Python Tutorial

## Note Specific

- [Concepts](lessons/concepts/readme.md)
- [Basics](lessons/basics/readme.md)

---

## Core Concepts

- [Overview](lessons/overview/readme.md)
- [Syntax](lessons/syntax/readme.md)
- [Variables](lessons/variables/readme.md)
- [Input/Output](lessons/input_output/readme.md)
- [Scope and Access](lessons/scope/readme.md)

---

## Operators

- [Arithmetic Operators](lessons/arithmetic_operators/readme.md)
- [Assignment Operators](lessons/assignment_operators/readme.md)
- [Comparison Operators](lessons/comparison_operators/readme.md)
- [Logical Operators](lessons/logical_operators/readme.md)
- [Bitwise Operators](lessons/bitwise_operators/readme.md)
- [Membership Operators](lessons/membership_operators/readme.md)
- [Identity Operators](lessons/identity_operators/readme.md)

---

## Control Flow

- [Conditional Statements](lessons/conditional_statements/readme.md)
- [Loops](lessons/loops/readme.md)
- [Comprehensions](lessons/comprehensions/readme.md)
- [Pattern Matching (`match` / `case`)](lessons/pattern_matching/readme.md)

---

## [Data](lessons/data/readme.md)

### [Datatypes](lessons/datatypes/readme.md)

- Numeric: [Integer](lessons/integer/readme.md), [Float](lessons/float/readme.md), [Complex](lessons/complex/readme.md)
- String: [String](lessons/string/readme.md)
- Boolean: [Boolean](lessons/boolean/readme.md)
- Collections: [List](lessons/list/readme.md), [Tuples](lessons/tuples/readme.md), [Set](lessons/set/readme.md), [Dictionary](lessons/dictionary/readme.md)
- Binary: [Bytes](lessons/bytes/readme.md), [Bytearray](lessons/bytearray/readme.md), [Memoryview](lessons/memoryview/readme.md)

### [Collection Module](modules/collection_module/overview/readme.md)
- [Counter](modules/collection_module/counter/readme.md)
- [OrderedDict](modules/collection_module/ordereddict/readme.md)
- [defaultdict](modules/collection_module/defaultdict/readme.md)
- [deque](modules/collection_module/deque/readme.md)
- [ChainMap](modules/collection_module/chainmap/readme.md)
- [namedtuple](modules/collection_module/namedtuple/readme.md)

### [String](lessons/string/readme.md)

- [String Operations](lessons/string_operations/readme.md)
- [String Methods ](lessons/string_methods/readme.md)
- Regular Expressions: [re](modules/re/readme.md)

### [Input/Output Functions](lessons/input_output_functions/readme.md)

- [High Level Input/Output](lessons/high_level_input_output/readme.md)
- [Low level Input/Output](lessons/low_level_input_output/readme.md)
- [Data Serialization](lessons/data_serialization/readme.md): `pickle`, `json`

---

## [Functions](lessons/functions/readme.md)

- [Function Definition & Call](lessons/function_definition_call/readme.md)
- [Arguments (Positional, Keyword, Default, `-args`, `--kwargs`)](lessons/arguments/readme.md)
- [Return Statement](lessons/return_statement/readme.md)
- [Lambda Functions](lessons/lambda_functions/readme.md)
- [Recursion](lessons/recursion/readme.md)
- [Decorators](lessons/decorators/readme.md)
- [Closures](lessons/closures/readme.md)
- [Built-in Higher Order Functions (`map`, `filter`, `reduce`)](lessons/higher_order_functions/readme.md)
- [Anonymous Functions](lessons/anonymous_functions/readme.md)
- [Type Hinting (`typing` module)](lessons/type_hinting/readme.md)

---

## [OOPS](lessons/oop/readme.md)

- [Class and Objects](lessons/class_and_object/readme.md)
- [Encapsulation](lessons/encapsulation/readme.md)
- [Abstraction](lessons/abstraction/readme.md)
- [Inheritance](lessons/inheritance/readme.md) (Single, Multiple, Multilevel, Hierarchical)
- [Polymorphism](lessons/polymorphism/readme.md) (Overloading, Overriding)
- [Constructors and Destructors](lessons/constructors_destructors/readme.md)
- [self and super](lessons/self_and_super/readme.md)
- [Access Modifiers](lessons/access_modifiers/readme.md)
- [Methods Types](lessons/method_types/readme.md)

  - [Instance method](lessons/instant_methods/readme.md)
  - [Class method](lessons/class_methods/readme.md)
  - [Static method](lessons/static_methods/readme.md)
- [Variable Types](lessons/variable_types/readme.md)

  - [Instance Variables](lessons/instant_variable/readme.md)
  - [Local Variables](lessons/local_variable/readme.md)
  - [Global Variables](lessons/global_variable/readme.md)
  - [Class Variable](lessons/class_variable/readme.md)
- [Property Decorator](lessons/property_decorator/readme.md)
- [Magic Methods](lessons/magic_methods/readme.md)
- [MRO and Conflict Resolution](lessons/mro_and_conflicts/readme.md)
- [Method Overloading](lessons/method_overloading/readme.md)
- [Multiple Dispatch](lessons/multiple_dispatch/readme.md)
- [Abstract Classes and Interfaces](lessons/abstract_classes_interfaces/readme.md)

  - [Abstract Class](lesson/abstract_class/readme.md)
  - [Interface](lessons/interface/readme.md)
  - [ABC Class](lessons/abc_class/readme.md)
- [OOP Cheatsheet](lessons/oops_cheatsheet/readme.md)

---

## Concurrency & Parallelism

- Multithreading: [Multithreading](modules/threading/multithreading/readme.md)
- Multiprocessing: [Multiprocessing](modules/multiprocessing/readme.md)
- Asynchronous Processing: [Asyncio](modules/asyncio/readme.md), [Concurrent Futures](modules/concurrent_futures/readme.md)
- Thread Synchronization: [Threading](modules/threading/readme.md) ([Lock](modules/threading/lock/readme.md), [RLock](modules/threading/rlock/readme.md), [Semaphore](modules/threading/semaphore/readme.md))

---

## File Handling

- File Modes (`r`, `w`, `a`, `rb`, `wb`)
- Reading & Writing
- File Iteration
- Context Managers (`with` statement)
- File System Operations (`os`, `shutil`, `pathlib`)

---

## Exception Handling

- [Overview](lessons/exception_handling/overview/readme.md)
- [Syntax](lessons/exception_handling/syntax/readme.md)
- Raising Exceptions (`raise`): [raise](lessons/exception_handling/raise/readme.md)
- [Custom Exceptions](lessons/exception_handling/custom_exception/readme.md)
- [Exception Hierarchy](lessons/exception_handling/exception_hierarchy/readme.md)

---

## Iterators & Generators

- Iterators (`iter()`, `next()`)
- Generators (`yield`)
- Generator Expressions

---

## Memory Management

- Garbage Collection (`gc` module)
- Reference Counting
- Weak References (`weakref` module)

---

## Environment & OS Interaction

- `os` module
- `sys` module
- Environment Variables
- Command-line Arguments (`argparse`)

---

## Date & Time

- `datetime` module
- `time` module
- `calendar` module

---

## Networking

- [Sockets](modules/sockets/readme.md)
- HTTP Requests: [requests](modules/requests/readme.md)
- FTP: [FTP](modules/ftp/readme.md)
- SMTP: [SMTP](modules/smtp/readme.md)

---

## Database Handling

- SQLite: [sqlite3](modules/sqlite3/readme.md) 
- MySQL: [mysql.connector](modules/mysql_connector/readme.md), [SQLAlchemy](library/sqlalchemy/readme.md)
- PostgreSQL: [psycopg2](modules/psycopg2/readme.md)
<!-- - ORM Basics -->

---

## Testing & Debugging

- [unittest](frameworks/unittest/readme.md)
- [pytest](packages/pytest/readme.md)
- Logging: [logging](modules/logging/readme.md)
- Debugging: [pdb](modules/pdb/readme.md)

---

## [Packages and Frameworks](lessons/packages_framework/readme.md)

###  Web Development
- [Django](frameworks/django/readme.md)
- [Flask](frameworks/flask/readme.md)
- [Streamlit](frameworks/streamlit/readme.md)

### GUI Creation
- [Tkinter](frameworks/tkinter/readme.md)
- [PySimpleGUI](frameworks/pysimplegui/readme.md)

### Data Analysis
- [Pandas](libraries/pandas/readme.md)
- [Numpy](packages/numpy/readme.md)

### Visualization
- [MatplotLib](packages/matplotlib/readme.md)
- [Seaborn](packages/seaborn/readme.md)

### Database Connector 
- [Pyodbc](packages/pyodbc/readme.md)
- Sqlite:[sqlite3](modules/sqlite3/readme.md) 
- MySQL: [mysql.connector](modules/mysql_connector/readme.md), 
- PostgreSQL: [psycopg2](modules/psycopg2/readme.md)
- ORM: [SQLAlchemy](library/sqlalchemy/readme.md)

### Machine Learning
- [Scikit Learn](packages/sklearn/readme.md)
- [Tensorflow](https://github.com/AkashDas253/Tutorial_Tensorflow)

### Web Scraping
- [BeautifulSoup](packages/beautifulsoup/readme.md)

### Automation
- [Selenium (PythonAPI)](https://github.com/AkashDas253/Tutorial_Tools/blob/master/tools/selenium/lessons/python_support/readme.md)

### Testing
- [unittest](frameworks/unittest/readme.md)
- [pytest](packages/pytest/readme.md)

---