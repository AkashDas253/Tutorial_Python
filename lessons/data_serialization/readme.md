## Data Serialization in Python

### What is Data Serialization?

**Data serialization** is the process of converting **Python objects** into a format that can be **stored** (e.g., in a file or memory) or **transmitted** (e.g., over a network), and later **reconstructed** (deserialized) back to the original object.

---

## Why Serialization is Needed

* To **save program state** (e.g., saving a model, config, or cache)
* To **send data over network** (e.g., APIs, sockets)
* To **store structured data** in text/binary format (e.g., in databases or files)
* For **interoperability** between systems and languages

---

## Common Serialization Formats in Python

| Format      | Module      | Format Type | Use Case                               |
| ----------- | ----------- | ----------- | -------------------------------------- |
| JSON        | `json`      | Text        | Web APIs, config files                 |
| Pickle      | `pickle`    | Binary      | Python object storage (not cross-lang) |
| Marshal     | `marshal`   | Binary      | Internal Python use                    |
| CSV         | `csv`       | Text        | Tabular data                           |
| YAML        | `pyyaml`    | Text        | Config files (human-readable)          |
| XML         | `xml.etree` | Text        | Legacy and structured documents        |
| MessagePack | `msgpack`   | Binary      | Efficient cross-language data sharing  |

---

## JSON Serialization

### Features

* Human-readable, language-independent
* Supports basic data types: `dict`, `list`, `str`, `int`, `float`, `bool`, `None`

### Module: `json`

### Functions

```python
import json

# Serialize to JSON string
json_str = json.dumps(data)

# Write JSON to file
json.dump(data, file)

# Deserialize from JSON string
data = json.loads(json_str)

# Read JSON from file
data = json.load(file)
```

### Use Case:

```python
data = {"name": "Alice", "age": 25}
with open("data.json", "w") as f:
    json.dump(data, f)

with open("data.json") as f:
    loaded = json.load(f)
```

---

## Pickle Serialization

### Features

* Stores **any Python object**, including custom classes
* Not human-readable
* Not secure for untrusted input

### Module: `pickle`

### Functions

```python
import pickle

# Serialize to bytes
byte_data = pickle.dumps(obj)

# Write to file
pickle.dump(obj, file)

# Deserialize from bytes
obj = pickle.loads(byte_data)

# Read from file
obj = pickle.load(file)
```

### Use Case:

```python
data = {"id": 101, "status": True}
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

with open("data.pkl", "rb") as f:
    loaded = pickle.load(f)
```

---

## Marshal

### Use

* For internal Python use (e.g., `.pyc` files)
* Faster but less portable and version-sensitive

```python
import marshal

data = {"x": 5}
bytes_data = marshal.dumps(data)
restored = marshal.loads(bytes_data)
```

---

## CSV (Comma-Separated Values)

### Module: `csv`

### Use Case:

```python
import csv

# Write
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 30])

# Read
with open("data.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

---

## YAML (Using `pyyaml`)

### Features

* Human-readable
* Used in configs (e.g., Kubernetes, GitHub Actions)

```python
import yaml

# Serialize
yaml_str = yaml.dump(data)

# Deserialize
data = yaml.load(yaml_str, Loader=yaml.FullLoader)
```

---

## XML

### Module: `xml.etree.ElementTree`

```python
import xml.etree.ElementTree as ET

# Create XML
root = ET.Element("person")
ET.SubElement(root, "name").text = "Alice"
tree = ET.ElementTree(root)
tree.write("person.xml")

# Parse XML
tree = ET.parse("person.xml")
root = tree.getroot()
```

---

## MessagePack

### Efficient binary serialization (cross-language)

```python
import msgpack

packed = msgpack.packb(data)
unpacked = msgpack.unpackb(packed)
```

---

## Security Warning

* **Never unpickle data from untrusted sources** — it can execute arbitrary code.
* Prefer **JSON** for safe and readable serialization.

---

## Summary Table

| Feature                   | JSON | Pickle | Marshal | CSV | YAML | XML | MessagePack |
| ------------------------- | ---- | ------ | ------- | --- | ---- | --- | ----------- |
| Readable                  | ✅    | ❌      | ❌       | ✅   | ✅    | ✅   | ❌           |
| Supports all Python types | ❌    | ✅      | ❌       | ❌   | ✅    | ❌   | ✅           |
| Secure                    | ✅    | ❌      | ❌       | ✅   | ✅    | ✅   | ✅           |
| Cross-lang                | ✅    | ❌      | ❌       | ✅   | ✅    | ✅   | ✅           |

---
