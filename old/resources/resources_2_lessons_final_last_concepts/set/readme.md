
## Sets:

### In-Depth Note on Python Sets

#### Properties of Sets

- **Unordered**: Sets are unordered collections of items.
- **Mutable**: Sets can be changed after creation.
- **Unique Elements**: Sets do not allow duplicate elements.
- **Immutable Elements**: Elements in a set must be immutable (e.g., strings, numbers, tuples).

#### Creating a Set

1. **Empty Set**
   ```python
   my_set = set()
   ```

2. **Set with Initial Values**
   ```python
   my_set = {1, 2, 3, 4, 5}
   ```

#### Adding Elements

1. **Using `add` Method**
   ```python
   my_set.add(6)
   ```

2. **Using `update` Method** (adds multiple elements)
   ```python
   my_set.update([7, 8, 9])
   ```

#### Removing Elements

1. **Using `remove` Method** (raises KeyError if element not found)
   ```python
   my_set.remove(3)
   ```

2. **Using `discard` Method** (does not raise an error if element not found)
   ```python
   my_set.discard(4)
   ```

3. **Using `pop` Method** (removes and returns an arbitrary element)
   ```python
   element = my_set.pop()
   ```

4. **Using `clear` Method** (removes all elements)
   ```python
   my_set.clear()
   ```

#### Set Operations

1. **Union**
   ```python
   set1 = {1, 2, 3}
   set2 = {3, 4, 5}
   union_set = set1 | set2  # or set1.union(set2)
   ```

2. **Intersection**
   ```python
   intersection_set = set1 & set2  # or set1.intersection(set2)
   ```

3. **Difference**
   ```python
   difference_set = set1 - set2  # or set1.difference(set2)
   ```

4. **Symmetric Difference**
   ```python
   sym_diff_set = set1 ^ set2  # or set1.symmetric_difference(set2)
   ```

#### Set Methods

1. **`add` Method**: Adds an element to the set
   ```python
   my_set.add(10)
   ```

2. **`update` Method**: Adds multiple elements to the set
   ```python
   my_set.update([11, 12])
   ```

3. **`remove` Method**: Removes an element from the set (raises KeyError if not found)
   ```python
   my_set.remove(10)
   ```

4. **`discard` Method**: Removes an element from the set (does not raise an error if not found)
   ```python
   my_set.discard(11)
   ```

5. **`pop` Method**: Removes and returns an arbitrary element from the set
   ```python
   element = my_set.pop()
   ```

6. **`clear` Method**: Removes all elements from the set
   ```python
   my_set.clear()
   ```

7. **`union` Method**: Returns the union of sets
   ```python
   union_set = set1.union(set2)
   ```

8. **`intersection` Method**: Returns the intersection of sets
   ```python
   intersection_set = set1.intersection(set2)
   ```

9. **`difference` Method**: Returns the difference of sets
   ```python
   difference_set = set1.difference(set2)
   ```

10. **`symmetric_difference` Method**: Returns the symmetric difference of sets
    ```python
    sym_diff_set = set1.symmetric_difference(set2)
    ```

11. **`issubset` Method**: Checks if one set is a subset of another
    ```python
    is_subset = set1.issubset(set2)
    ```

12. **`issuperset` Method**: Checks if one set is a superset of another
    ```python
    is_superset = set1.issuperset(set2)
    ```

13. **`isdisjoint` Method**: Checks if two sets have no elements in common
    ```python
    is_disjoint = set1.isdisjoint(set2)
    ```

#### Set Comprehension

1. **Basic Set Comprehension**
   ```python
   squared_set = {x*x for x in range(10)}
   ```

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
