## **Bytearray in Python**  

A `bytearray` is a **mutable sequence** of integers (0–255), used for handling binary data that needs modification.

---

## **1. Creating a `bytearray`**  

### **Using `bytearray()` Constructor**
```python
ba = bytearray([65, 66, 67])  # ASCII for A, B, C
print(ba)  # Output: bytearray(b'ABC')
```

### **Using `b''` with `bytearray()`**
```python
ba = bytearray(b"Hello")
print(ba)  # Output: bytearray(b'Hello')
```

### **Using `bytearray(size)` (Zero-filled)**
```python
ba = bytearray(5)  # Creates b'\x00\x00\x00\x00\x00'
print(ba)
```

---

## **2. Modifying `bytearray` (Mutable)**  

### **Changing Elements**
```python
ba = bytearray(b"Hello")
ba[0] = 72  # ASCII for 'H'
ba[1] = 101  # ASCII for 'e'
print(ba)  # Output: bytearray(b'Hello')
```

### **Appending Bytes**
```python
ba.append(33)  # ASCII for '!'
print(ba)  # Output: bytearray(b'Hello!')
```

### **Extending `bytearray`**
```python
ba.extend(b" World")
print(ba)  # Output: bytearray(b'Hello! World')
```

### **Slicing (Modify Multiple Bytes)**
```python
ba[6:11] = b"Python"
print(ba)  # Output: bytearray(b'Hello!Python')
```

---

## **3. Converting Between Data Types**  

| **Conversion** | **Method** |
|--------------|-----------|
| String → Bytearray | `bytearray("Hello", "utf-8")` |
| Bytearray → String | `bytearray.decode("utf-8")` |
| List → Bytearray | `bytearray([65, 66, 67])` |
| Bytearray → List | `list(bytearray(b'ABC'))` |

### **Example**
```python
ba = bytearray("Hello", "utf-8")  # String to bytearray
s = ba.decode("utf-8")  # Bytearray to string
print(s)  # Output: Hello
```

---

## **4. File Handling with `bytearray`**  

### **Writing Binary Data**
```python
with open("file.bin", "wb") as file:
    file.write(bytearray(b"Binary Data"))
```

### **Reading Binary Data**
```python
with open("file.bin", "rb") as file:
    ba = bytearray(file.read())
    print(ba)  # Output: bytearray(b'Binary Data')
```

---

## **5. Using `bytearray.fromhex()`**
```python
ba = bytearray.fromhex("48656c6c6f")  # "Hello" in hex
print(ba)  # Output: bytearray(b'Hello')
```

---
