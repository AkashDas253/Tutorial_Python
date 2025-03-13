## **Input and Output in Python**  

Python provides built-in functions for **reading input** from the user and **displaying output** on the screen.

---

## **1. Input in Python (`input()`)**  

### **Syntax**
```python
variable = input("Prompt message")
```

### **Example**
```python
name = input("Enter your name: ")
print("Hello,", name)
```
📌 **`input()` always returns a string**, so type conversion is needed for numerical input.

### **Taking Integer or Float Input**
```python
age = int(input("Enter your age: "))  # Converts input to integer
height = float(input("Enter your height: "))  # Converts input to float
```

### **Multiple Inputs (Using `split()`)**
```python
x, y = input("Enter two numbers: ").split()
print("You entered:", x, "and", y)
```

### **List Input (Using `map()`)**
```python
numbers = list(map(int, input("Enter numbers: ").split()))
print("Numbers:", numbers)
```

---

## **2. Output in Python (`print()`)**  

### **Syntax**
```python
print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
```

### **Example**
```python
print("Hello, World!")  # Output: Hello, World!
```

### **Using `sep` (Separator)**
```python
print("Python", "is", "fun", sep="-")  # Output: Python-is-fun
```

### **Using `end` (Change Line Ending)**
```python
print("Hello", end=" ")
print("World!")  # Output: Hello World!
```

### **Printing Variables**
```python
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old.")  # Output: My name is Alice and I am 25 years old.
```

---

## **3. File Input and Output**  

### **Writing to a File**
```python
with open("file.txt", "w") as file:
    file.write("Hello, File!")
```

### **Reading from a File**
```python
with open("file.txt", "r") as file:
    content = file.read()
    print(content)  # Output: Hello, File!
```

---
