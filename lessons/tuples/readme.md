
## Tuples

### In-Depth Note on Python Tuples

#### Properties of Tuples

- **Ordered**: Tuples maintain the order of elements.
- **Immutable**: Once created, the elements of a tuple cannot be changed.
- **Indexed**: Elements can be accessed using indices.
- **Heterogeneous**: Tuples can contain elements of different data types.
- **Hashable**: Tuples can be used as keys in dictionaries if they contain only hashable types.

#### Creating Tuples

1. **Empty Tuple**
   ```python
   my_tuple = ()
   ```

2. **Tuple with One Element** (note the comma)
   ```python
   my_tuple = (element,)
   ```

3. **Tuple with Multiple Elements**
   ```python
   my_tuple = (element1, element2, element3)
   ```

4. **Tuple Without Parentheses** (tuple packing)
   ```python
   my_tuple = element1, element2, element3
   ```

5. **Using `tuple` Constructor**
   ```python
   my_tuple = tuple([element1, element2, element3])
   ```

#### Accessing Elements

1. **Using Indexing**
   ```python
   element = my_tuple[0]
   ```

2. **Using Negative Indexing**
   ```python
   element = my_tuple[-1]
   ```

3. **Using Slicing**
   ```python
   sub_tuple = my_tuple[1:3]
   ```

#### Tuple Operations

1. **Concatenation**
   ```python
   new_tuple = my_tuple1 + my_tuple2
   ```

2. **Repetition**
   ```python
   repeated_tuple = my_tuple * 3
   ```

3. **Membership Test**
   ```python
   if element in my_tuple:
       # Code to execute if element is in my_tuple
   ```

4. **Iterating Over Elements**
   ```python
   for element in my_tuple:
       print(element)
   ```

#### Tuple Methods

1. **`count` Method**: Returns the number of occurrences of a specified value
   ```python
   count = my_tuple.count(element)
   ```

2. **`index` Method**: Returns the index of the first occurrence of a specified value
   ```python
   index = my_tuple.index(element)
   ```

#### Unpacking Tuples

1. **Basic Unpacking**
   ```python
   a, b, c = my_tuple
   ```

2. **Unpacking with `*` Operator**
   ```python
   a, *b, c = my_tuple
   ```

#### Nested Tuples

1. **Creating Nested Tuples**
   ```python
   nested_tuple = (element1, (element2, element3), element4)
   ```

2. **Accessing Elements in Nested Tuples**
   ```python
   inner_element = nested_tuple[1][0]
   ```

#### Tuple Comprehension (Not Directly Supported, Use Generator Expression)

1. **Using Generator Expression**
   ```python
   my_tuple = tuple(x*x for x in range(5))
   ```

### Tuple Methods

- `tuple.count(value)` - Returns the number of times a specified value occurs in a tuple. `value` is the element to count.
- `tuple.index(value, start=0, end=len(tuple))` - Searches the tuple for a specified value and returns the position of where it was found. `value` is the element to search for, `start` and `end` specify the range to search within.