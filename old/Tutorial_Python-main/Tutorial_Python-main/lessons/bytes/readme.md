## **Bytes in Python**  

The `bytes` type represents **immutable sequences** of integers in the range **0–255**, typically used for binary data like files, network communication, or encryption.  

---

## **1. Creating Bytes Objects**  

### **Using `b''` or `B''`**
```python
b1 = b"Hello"
print(b1)  # Output: b'Hello'
```

### **Using `bytes()` Constructor**
```python
b2 = bytes([65, 66, 67])  # ASCII for A, B, C
print(b2)  # Output: b'ABC'
```

### **Using `encode()` Method**
```python
s = "Hello"
b3 = s.encode("utf-8")  # Convert string to bytes
print(b3)  # Output: b'Hello'
```

---

## **2. Accessing Bytes Elements**  
Each byte is an integer (0–255).  
```python
b = b"Hello"
print(b[0])  # Output: 72 (ASCII of 'H')
print(b[:2])  # Output: b'He'
```

---

## **3. Modifying Bytes (Immutable)**  
Bytes **cannot** be modified directly.  
```python
b = b"Hello"
# b[0] = 65  # ❌ TypeError: 'bytes' object does not support item assignment
```
Use `bytearray` for mutable bytes.

---

## **4. `bytearray` (Mutable Bytes Sequence)**  
```python
ba = bytearray([65, 66, 67])
ba[0] = 97  # Modify first byte
print(ba)  # Output: bytearray(b'aBC')
```

---

## **5. Converting Between Data Types**  

| **Conversion** | **Method** |
|--------------|-----------|
| String → Bytes | `string.encode("utf-8")` |
| Bytes → String | `bytes.decode("utf-8")` |
| List → Bytes | `bytes([65, 66, 67])` |
| Bytes → List | `list(b'ABC')` |

### **Example**
```python
b = b"Hello"
s = b.decode("utf-8")  # Convert bytes to string
print(s)  # Output: Hello
```

---

## **6. Reading & Writing Bytes in Files**  
### **Writing Binary Data**
```python
with open("file.bin", "wb") as file:
    file.write(b"Binary Data")
```
### **Reading Binary Data**
```python
with open("file.bin", "rb") as file:
    content = file.read()
    print(content)  # Output: b'Binary Data'
```

---

## **7. Using `bytes.fromhex()` for Hexadecimal Input**
```python
b = bytes.fromhex("48656c6c6f")  # "Hello" in hex
print(b)  # Output: b'Hello'
```

---
