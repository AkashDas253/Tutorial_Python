## Element Metadata in PySimpleGUI

Element metadata allows developers to associate **custom data or tags** with elements, useful for maintaining context, managing state, or attaching additional information without using global variables or complex structures.

---

### Purpose of Metadata

- Attach application-specific data to GUI elements
- Simplify event handling by storing state or identity directly in the element
- Avoid external mapping or lookup structures
- Great for dynamically generated UIs or reusable components

---

### Attribute: `metadata`

Every PySimpleGUI Element supports the `metadata` parameter.

```python
sg.Button("Click Me", key="-BTN-", metadata={"id": 101, "type": "submit"})
```

You can also assign metadata after creation:

```python
button = sg.Button("Click Me", key="-BTN-")
button.metadata = "custom_info"
```

---

### Accessing Metadata

During event handling, access metadata via the element instance:

```python
event, values = window.read()
if event == "-BTN-":
    element = window["-BTN-"]
    print(element.metadata)
```

---

### Use Cases

| Use Case                         | Example                                        |
|----------------------------------|------------------------------------------------|
| Identify source of action        | Buttons with metadata indicating row/column   |
| Tag elements with context        | Store parent container ID                     |
| Hold intermediate state          | Flag if an item is selected or disabled       |
| Associate external data objects  | Link to a database row or JSON object         |

---

### Data Types Allowed

- Any Python object: int, str, list, dict, custom class instance
- Should be **serializable** or **self-contained**

---

### Best Practices

- Use descriptive metadata for debugging
- Avoid overloading metadata with too many responsibilities
- For complex apps, define metadata schemas or types

---
