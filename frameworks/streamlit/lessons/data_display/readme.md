# **Data Display** in Streamlit

Streamlit offers several built-in functions to **display, format, and interact with data** in various forms — ranging from simple text to complex DataFrames, JSON structures, tables, and metrics. This note outlines **all standard and advanced data display methods**, **parameter details**, and **when to use each**.

---

## Overview of Data Display Functions

| Function            | Purpose                                      | Typical Use Case                          |
|---------------------|----------------------------------------------|--------------------------------------------|
| `st.write()`        | Generic display method for various objects   | Quick display of any object (text, data)   |
| `st.dataframe()`    | Interactive table view of a DataFrame        | Scrollable, editable tabular data view     |
| `st.table()`        | Static table with styled formatting          | Simple static presentation of data         |
| `st.metric()`       | Display KPI metrics with optional delta      | Dashboard-style summary metrics            |
| `st.json()`         | Render formatted JSON data                   | Display dictionaries or structured results |
| `st.code()`         | Display syntax-highlighted code              | Show code snippets or logs                 |
| `st.text()`         | Render plain preformatted text               | Simple logs, fixed-width output            |
| `st.markdown()`     | Rich text with Markdown formatting           | Show headings, links, bold, etc.           |

---

## `st.write()`

- **Purpose**: Most flexible function — automatically detects and displays most object types (text, DataFrames, plots, etc.)

### Syntax:
```python
st.write(*args, unsafe_allow_html=False)
```

| Parameter           | Description                                           |
|---------------------|-------------------------------------------------------|
| `*args`             | Object(s) to display                                  |
| `unsafe_allow_html` | Whether to render raw HTML tags (default: `False`)   |

**Notes**:
- Handles strings, numbers, DataFrames, matplotlib, altair, plotly, etc.
- Great for prototyping.

---

## `st.dataframe()`

- **Purpose**: Display a **scrollable, interactive** DataFrame.
- **Supports**: Sorting, resizing, highlighting, column formatting.

### Syntax:
```python
st.dataframe(data, width=None, height=None, use_container_width=False, hide_index=False, column_order=None, column_config=None)
```

| Parameter             | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| `data`                | DataFrame, list, dict, or similar data                        |
| `width`, `height`     | Dimensions of the data widget                                 |
| `use_container_width` | Stretch to full container width (`False` by default)         |
| `hide_index`          | Whether to show index column (`False` by default)            |
| `column_order`        | Custom order of columns                                       |
| `column_config`       | Dict of column formatting options                             |

---

## `st.table()`

- **Purpose**: Display a **static** non-interactive table.
- **Supports**: Pandas DataFrame, list of lists, dicts, etc.

### Syntax:
```python
st.table(data)
```

| Parameter | Description                |
|-----------|----------------------------|
| `data`    | Tabular data to display     |

**Differences vs `st.dataframe()`**:
- No interactivity (no sorting, no resizing)
- Better for displaying final results or formatted text tables

---

## `st.metric()`

- **Purpose**: Display a **numeric value with an optional delta**, often used in dashboards.

### Syntax:
```python
st.metric(label, value, delta=None, delta_color="normal", help=None, label_visibility="visible")
```

| Parameter         | Description                                               |
|-------------------|-----------------------------------------------------------|
| `label`           | Metric label (e.g., "Revenue")                            |
| `value`           | Main value to show (numeric or string)                    |
| `delta`           | Optional value to compare (numeric)                       |
| `delta_color`     | Color theme: `"normal"` (default), `"inverse"`, `"off"`  |
| `help`            | Tooltip help text                                         |
| `label_visibility`| `"visible"`, `"hidden"`, or `"collapsed"`                |

---

## `st.json()`

- **Purpose**: Display structured JSON (dictionary, list) with formatting and indentation.

### Syntax:
```python
st.json(body, expanded=True)
```

| Parameter   | Description                                |
|-------------|--------------------------------------------|
| `body`      | JSON-like dict, list, or string             |
| `expanded`  | Whether to show all items by default        |

---

## `st.code()`

- **Purpose**: Display syntax-highlighted code block.

### Syntax:
```python
st.code(body, language="python", line_numbers=False)
```

| Parameter     | Description                                  |
|---------------|----------------------------------------------|
| `body`        | Code as string                               |
| `language`    | Code language (for syntax highlighting)      |
| `line_numbers`| Show line numbers (`False` by default)       |

---

## `st.text()`

- **Purpose**: Display fixed-width plain text (no formatting or highlighting).

### Syntax:
```python
st.text(body)
```

| Parameter | Description                    |
|-----------|--------------------------------|
| `body`    | The plain text to be displayed |

---

## `st.markdown()`

- **Purpose**: Render Markdown-formatted text (headings, bold, links, etc.)

### Syntax:
```python
st.markdown(body, unsafe_allow_html=False, help=None)
```

| Parameter           | Description                                              |
|---------------------|----------------------------------------------------------|
| `body`              | Markdown-formatted string                               |
| `unsafe_allow_html` | Allow raw HTML rendering (disabled by default)          |
| `help`              | Tooltip help                                             |

---

## Choosing the Right Function

| Requirement                                | Use Function         |
|--------------------------------------------|----------------------|
| Display any object                         | `st.write()`         |
| Scrollable and sortable table              | `st.dataframe()`     |
| Static formatted table                     | `st.table()`         |
| Show KPI with delta                        | `st.metric()`        |
| View JSON-like structured data             | `st.json()`          |
| Show source code                           | `st.code()`          |
| Display preformatted plain text            | `st.text()`          |
| Render rich text (Markdown, HTML)          | `st.markdown()`      |

---

## 📝 Example Combination

```python
import streamlit as st
import pandas as pd

# Sample data
data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

st.title("Data Display Demo")
st.write("Here is a sample DataFrame:")
st.dataframe(data)

st.metric(label="Revenue", value="$10,000", delta="+5%")

st.json({"name": "Alice", "age": 30})

st.code("print('Hello World')", language="python")
```

---
