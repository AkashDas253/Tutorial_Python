# **Built-in Chart Functions** in Streamlit

Streamlit provides **high-level chart functions** that make it easy to create common visualizations directly from data structures like Pandas DataFrames and NumPy arrays.

---

## 🔹 Overview of Built-in Chart Functions

| Function             | Description                          | Data Types Supported                  |
|----------------------|--------------------------------------|----------------------------------------|
| `st.line_chart()`    | Line graph for trend analysis        | Pandas DataFrame, Series, NumPy array  |
| `st.area_chart()`    | Area graph for cumulative trends     | Pandas DataFrame, Series, NumPy array  |
| `st.bar_chart()`     | Bar chart for category comparison    | Pandas DataFrame, Series, NumPy array  |
| `st.map()`           | Plots data points on a map           | DataFrame with `lat` and `lon` columns |
| `st.metric()`        | Displays a KPI/metric with delta     | Strings/numbers                        |

---

## 🔸 `st.line_chart()`

- **Purpose**: Display trends over time or continuous variables.

### ✅ Syntax:
```python
st.line_chart(data, x=None, y=None, width=0, height=0, use_container_width=True)
```

| Parameter              | Description                                              |
|------------------------|----------------------------------------------------------|
| `data`                 | DataFrame, Series, or array to plot                      |
| `x`, `y`               | Optional column names                                    |
| `width`, `height`     | Size in pixels                                           |
| `use_container_width` | Stretch to width of container (`True` by default)        |

---

## 🔸 `st.area_chart()`

- **Purpose**: Like a line chart, but fills the area beneath the lines.

### ✅ Syntax:
```python
st.area_chart(data, x=None, y=None, width=0, height=0, use_container_width=True)
```

---

## 🔸 `st.bar_chart()`

- **Purpose**: Compare values across discrete categories.

### ✅ Syntax:
```python
st.bar_chart(data, x=None, y=None, width=0, height=0, use_container_width=True)
```

---

## 🔸 `st.map()`

- **Purpose**: Geospatial plotting using latitude and longitude.

### ✅ Syntax:
```python
st.map(data, zoom=10, use_container_width=True)
```

| Parameter              | Description                                      |
|------------------------|--------------------------------------------------|
| `data`                 | DataFrame with `lat` and `lon` columns           |
| `zoom`                 | Initial zoom level (default: 10)                 |
| `use_container_width` | Stretch to container (default: `True`)           |

---

## 🔸 `st.metric()`

- **Purpose**: Displays key performance indicators (KPIs).

### ✅ Syntax:
```python
st.metric(label, value, delta=None, delta_color="normal", help=None)
```

| Parameter      | Description                                               |
|----------------|-----------------------------------------------------------|
| `label`        | Label shown above the metric                              |
| `value`        | Current value (number or string)                          |
| `delta`        | Change indicator (optional)                               |
| `delta_color`  | `"normal"`, `"inverse"`, `"off"`                          |
| `help`         | Tooltip message                                           |

---

## 🧩 Example:

```python
import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)

# Metric
st.metric("Revenue", "$120K", "+5.6%")

# Map
df_map = pd.DataFrame({
    'lat': [37.76, 37.77],
    'lon': [-122.4, -122.42]
})
st.map(df_map)
```

---
