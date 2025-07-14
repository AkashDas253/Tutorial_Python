## **MemoryView in Python**  

A `memoryview` provides a way to access the memory of an object **without copying** it. It works with **byte-like objects** (`bytes`, `bytearray`) and allows efficient slicing and manipulation.

---

## **1. Creating a `memoryview`**  

### **Using `memoryview()` on a Byte-like Object**
```python
ba = bytearray("Hello", "utf-8")  # Create a bytearray
mv = memoryview(ba)  # Create a memoryview
print(mv)  # Output: <memory at 0x...>
```

---

## **2. Accessing Elements**  
A `memoryview` behaves like a sequence of bytes.

```python
print(mv[0])  # Output: 72 (ASCII of 'H')
print(mv[:5].tobytes())  # Output: b'Hello'
```

---

## **3. Modifying Bytes in `memoryview`**  
Unlike `bytes`, `memoryview` allows modification when used with `bytearray`.

```python
mv[0] = 74  # ASCII for 'J'
print(ba)  # Output: bytearray(b'Jello')
```
*Note:* `bytes` are immutable, so modifying `memoryview(bytes_object)` raises an error.

---

## **4. Converting `memoryview` to Other Types**  

| **Conversion** | **Method** |
|--------------|-----------|
| MemoryView → Bytes | `mv.tobytes()` |
| MemoryView → List | `list(mv)` |

### **Example**
```python
print(mv.tobytes())  # Output: b'Jello'
print(list(mv[:5]))  # Output: [74, 101, 108, 108, 111]
```

---

## **5. Using `memoryview.cast()` for Type Conversion**  

### **Example: Casting to Different Formats**
```python
numbers = bytearray([1, 2, 3, 4])
mv = memoryview(numbers).cast("H")  # 'H' for 2-byte integers
print(mv.tolist())  # Output: [513, 1027] (Little-endian interpretation)
```

---

## **6. Working with Binary Files Using `memoryview`**
### **Writing to File Efficiently**
```python
with open("binary.bin", "wb") as file:
    file.write(memoryview(bytearray([65, 66, 67])))  # Writes b'ABC'
```

### **Reading from File**
```python
with open("binary.bin", "rb") as file:
    mv = memoryview(file.read())
    print(mv.tobytes())  # Output: b'ABC'
```

---
