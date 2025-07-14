## **Saving and Loading Data in NumPy**  

NumPy provides efficient methods for saving and loading arrays in various formats, optimizing performance and storage.

---

### **1. Saving and Loading in NumPy Format (`.npy`, `.npz`)**  

NumPy’s native formats store arrays efficiently while preserving `dtype`.  

| Method | Description | Syntax |
|--------|-------------|--------|
| **Save a single array (`.npy`)** | Stores one array in binary format. | `np.save('file.npy', arr)` |
| **Load a single array (`.npy`)** | Reads a saved NumPy array. | `arr = np.load('file.npy')` |
| **Save multiple arrays (`.npz`)** | Stores multiple arrays in a compressed file. | `np.savez('file.npz', arr1=arr1, arr2=arr2)` |
| **Load multiple arrays (`.npz`)** | Reads multiple saved arrays. | `data = np.load('file.npz')` <br> `arr1 = data['arr1']` |

---

### **2. Saving and Loading in Text Format (`.txt`, `.csv`)**  

Text formats allow human-readable storage but may increase file size.  

| Method | Description | Syntax |
|--------|-------------|--------|
| **Save as text (`.txt`, `.csv`)** | Writes array to a text file. | `np.savetxt('file.txt', arr, delimiter=',')` |
| **Load from text (`.txt`, `.csv`)** | Reads array from a text file. | `arr = np.loadtxt('file.txt', delimiter=',')` |

---

### **3. Saving and Loading Using Pickle (`.pkl`)**  

Pickle allows storing NumPy objects, preserving complex structures.  

| Method | Description | Syntax |
|--------|-------------|--------|
| **Save using Pickle** | Serializes and saves an array. | `import pickle` <br> `with open('file.pkl', 'wb') as f: pickle.dump(arr, f)` |
| **Load using Pickle** | Reads a Pickle file. | `with open('file.pkl', 'rb') as f: arr = pickle.load(f)` |

---

### **4. Saving and Loading Using Pandas (`.csv`, `.hdf5`)**  

Pandas provides additional flexibility for tabular data.  

| Method | Description | Syntax |
|--------|-------------|--------|
| **Save as CSV (`.csv`)** | Saves an array as a CSV file. | `pd.DataFrame(arr).to_csv('file.csv', index=False, header=False)` |
| **Load from CSV (`.csv`)** | Reads data into a NumPy array. | `arr = pd.read_csv('file.csv', header=None).values` |
| **Save as HDF5 (`.h5`)** | Stores large datasets efficiently. | `import h5py` <br> `with h5py.File('file.h5', 'w') as f: f.create_dataset('dataset', data=arr)` |
| **Load from HDF5 (`.h5`)** | Reads an HDF5 file. | `with h5py.File('file.h5', 'r') as f: arr = f['dataset'][:]` |

---

### **5. Using `memory-map` for Large Data (`.dat`)**  

Memory-mapped files allow efficient access to large datasets without loading them entirely into RAM.  

| Method | Description | Syntax |
|--------|-------------|--------|
| **Create a memory-mapped array** | Maps a file to memory for efficient access. | `mem_arr = np.memmap('file.dat', dtype=np.float32, mode='w+', shape=(rows, cols))` |
| **Read from memory-mapped array** | Accesses data efficiently. | `mem_arr = np.memmap('file.dat', dtype=np.float32, mode='r', shape=(rows, cols))` |

---

### **Summary**  
- **Use `.npy` and `.npz`** for fast, efficient storage of NumPy arrays.  
- **Use `.txt` or `.csv`** for human-readable text storage.  
- **Use Pickle (`.pkl`)** for serializing complex NumPy objects.  
- **Use Pandas (`.csv`, `.hdf5`)** for structured data storage.  
- **Use memory-mapped files (`.dat`)** for handling large datasets efficiently.