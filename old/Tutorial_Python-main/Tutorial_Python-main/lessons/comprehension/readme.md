## **Comprehensions in Python**  

Python **comprehensions** provide a concise way to create collections like lists, sets, and dictionaries using a single line of code.

---

## **List Comprehension**  

### **Syntax**  
```python
[expression for item in iterable if condition]
```

### **Example**  
```python
squares = [x**2 for x in range(5)]
print(squares)  # Output: [0, 1, 4, 9, 16]
```

### **With Condition**
```python
even_numbers = [x for x in range(10) if x % 2 == 0]
print(even_numbers)  # Output: [0, 2, 4, 6, 8]
```

### **Nested List Comprehension**  
```python
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(matrix)  # Output: [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
```

---

## **Set Comprehension**  

### **Syntax**  
```python
{expression for item in iterable if condition}
```

### **Example**  
```python
unique_squares = {x**2 for x in [1, 2, 2, 3, 4, 4]}
print(unique_squares)  # Output: {16, 1, 4, 9}
```

---

## **Dictionary Comprehension**  

### **Syntax**  
```python
{key_expression: value_expression for item in iterable if condition}
```

### **Example**  
```python
squares_dict = {x: x**2 for x in range(5)}
print(squares_dict)  # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

---

## **Generator Comprehension**  

### **Syntax**  
```python
(expression for item in iterable if condition)
```

### **Example**  
```python
gen = (x**2 for x in range(5))
print(next(gen))  # Output: 0
print(next(gen))  # Output: 1
```

---
