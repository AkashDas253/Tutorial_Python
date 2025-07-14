
## List:

### Properties and usage of Python Lists

#### Properties of Lists

- **Ordered**: Lists maintain the order of elements.
- **Mutable**: Lists can be changed after creation.
- **Indexed**: Lists are indexed by integers, starting from 0.
- **Heterogeneous**: Lists can contain elements of different data types.

#### Creating a List

1. **Empty List**
   ```python
   my_list = []
   ```

2. **List with Initial Values**
   ```python
   my_list = [1, 2, 3, "four", 5.0]
   ```

#### Accessing Elements

1. **Using Index**
   ```python
   element = my_list[0]  # Access the first element
   ```

2. **Using Negative Index**
   ```python
   element = my_list[-1]  # Access the last element
   ```

#### Adding Elements

1. **Using `append` Method** (adds an element to the end)
   ```python
   my_list.append("new_element")
   ```

2. **Using `insert` Method** (inserts an element at a specified position)
   ```python
   my_list.insert(1, "inserted_element")
   ```

3. **Using `extend` Method** (extends the list by appending elements from an iterable)
   ```python
   my_list.extend([6, 7, 8])
   ```

#### Removing Elements

1. **Using `remove` Method** (removes the first occurrence of a value)
   ```python
   my_list.remove("four")
   ```

2. **Using `pop` Method** (removes and returns the element at a specified position)
   ```python
   element = my_list.pop(2)
   ```

3. **Using `clear` Method** (removes all elements)
   ```python
   my_list.clear()
   ```

#### List Operations

1. **Checking if an Element Exists**
   ```python
   if "new_element" in my_list:
       # Code to execute if "new_element" exists in my_list
   ```

2. **Iterating Over Elements**
   ```python
   for element in my_list:
       print(element)
   ```

3. **List Comprehension**
   ```python
   squared_numbers = [x*x for x in range(5)]
   ```

4. **Slicing a List**
   ```python
   sub_list = my_list[1:3]  # Get elements from index 1 to 2
   ```

#### List Methods

1. **`index` Method**: Returns the index of the first occurrence of a value
   ```python
   index = my_list.index("new_element")
   ```

2. **`count` Method**: Returns the number of occurrences of a value
   ```python
   count = my_list.count(2)
   ```

3. **`sort` Method**: Sorts the list in ascending order
   ```python
   my_list.sort()
   ```

4. **`reverse` Method**: Reverses the elements of the list
   ```python
   my_list.reverse()
   ```

5. **`copy` Method**: Returns a shallow copy of the list
   ```python
   new_list = my_list.copy()
   ```

#### List Functions

1. **`len` Function**: Returns the number of elements in the list
   ```python
   length = len(my_list)
   ```

2. **`max` Function**: Returns the largest element in the list
   ```python
   maximum = max(my_list)
   ```

3. **`min` Function**: Returns the smallest element in the list
   ```python
   minimum = min(my_list)
   ```

4. **`sum` Function**: Returns the sum of all elements in the list
   ```python
   total = sum(my_list)
   ```

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
