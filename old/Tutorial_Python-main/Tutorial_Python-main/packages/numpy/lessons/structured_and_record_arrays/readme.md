## **Structured and Record Arrays in NumPy**  

NumPy provides **structured arrays** (also known as **record arrays**) to store heterogeneous data types in a single array, similar to a table with named columns.

---

### **Creating Structured Arrays**  

| Method | Description | Syntax |
|--------|-------------|--------|
| Using `dtype` | Define a structured array with field names and types. | `arr = np.array([(val1, val2)], dtype=[('name1', type1), ('name2', type2)])` |
| Using `np.zeros()` | Create an empty structured array. | `arr = np.zeros(size, dtype=[('name1', type1), ('name2', type2)])` |
| Using `np.dtype()` | Define a structured data type separately. | `dt = np.dtype([('name1', type1), ('name2', type2)])` |

---

### **Accessing Fields**  

| Operation | Description | Syntax |
|------------|-------------|--------|
| Access a field | Retrieve values from a specific field. | `arr['field_name']` |
| Access multiple fields | Retrieve multiple columns. | `arr[['field1', 'field2']]` |

---

### **Modifying Structured Arrays**  

| Operation | Description | Syntax |
|------------|-------------|--------|
| Assign new values | Modify values in a specific field. | `arr['field_name'] = value` |
| Add a new field | Use `np.lib.recfunctions.append_fields()`. | `new_arr = append_fields(arr, 'new_field', data, dtype)` |

---

### **Sorting and Searching in Structured Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `sort()` | Sort based on a field. | `arr.sort(order='field_name')` |
| `argsort()` | Get sorted indices based on a field. | `indices = arr.argsort(order='field_name')` |

---

### **Using Record Arrays**  
Record arrays (`np.recarray`) work like structured arrays but allow field access via attributes.

| Function | Description | Syntax |
|----------|-------------|--------|
| `fromarrays()` | Create a record array from multiple lists. | `rec = np.core.records.fromarrays([arr1, arr2], names='name1,name2')` |
| `fromrecords()` | Create a record array from tuples. | `rec = np.core.records.fromrecords(data, names='name1,name2')` |

---

### **Summary**  
- **Structured arrays** store multiple data types in a single array.  
- **Fields** can be accessed using names like dictionary keys.  
- **Sorting & searching** work by specifying field names.  
- **Record arrays** allow attribute-style access to fields.