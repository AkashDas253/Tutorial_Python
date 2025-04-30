# Comprehensive Note on **Custom Visualizations in Streamlit**

Streamlit allows developers to embed **custom visualizations** using popular external charting libraries for highly interactive and tailored visual outputs beyond the built-in chart functions.

---

## 🔹 Supported Libraries for Custom Visualizations

| Library           | Streamlit Function            | Description                                      |
|------------------|-------------------------------|--------------------------------------------------|
| **Matplotlib**    | `st.pyplot()`                 | Static 2D plots                                  |
| **Seaborn**       | `st.pyplot()`                 | Based on Matplotlib, high-level styling          |
| **Plotly**        | `st.plotly_chart()`           | Interactive charts with zoom and hover           |
| **Altair**        | `st.altair_chart()`           | Declarative grammar for building charts          |
| **Bokeh**         | `st.bokeh_chart()`            | Interactive plots for the web                    |
| **PyDeck**        | `st.pydeck_chart()`           | Geospatial 3D visualizations                     |
| **Vega-Lite JSON**| `st.vega_lite_chart()`        | Raw Vega-Lite spec usage                         |

---

## 🔸 `st.pyplot()`

- **Used With**: `matplotlib.pyplot`
- **Usage**: Displays a Matplotlib figure.

### ✅ Syntax:
```python
st.pyplot(fig=None, clear_figure=True, **kwargs)
```

| Parameter       | Description                                           |
|------------------|-------------------------------------------------------|
| `fig`            | Matplotlib figure object                             |
| `clear_figure`   | Whether to clear after rendering (`True` by default) |

---

## 🔸 `st.plotly_chart()`

- **Used With**: `plotly.graph_objects` or `plotly.express`

### ✅ Syntax:
```python
st.plotly_chart(fig, use_container_width=False, sharing="auto")
```

| Parameter            | Description                                           |
|----------------------|-------------------------------------------------------|
| `fig`                | Plotly figure object                                  |
| `use_container_width`| Stretch to full width                                |
| `sharing`            | For embedding behavior (`auto`, `streamlit`, `private`, `public`) |

---

## 🔸 `st.altair_chart()`

- **Used With**: `altair.Chart`

### ✅ Syntax:
```python
st.altair_chart(chart, use_container_width=False, theme=None)
```

| Parameter            | Description                                           |
|----------------------|-------------------------------------------------------|
| `chart`              | Altair chart object                                   |
| `theme`              | Use `"streamlit"` or `"none"`                         |

---

## 🔸 `st.bokeh_chart()`

- **Used With**: Bokeh figures

### ✅ Syntax:
```python
st.bokeh_chart(fig, use_container_width=False)
```

---

## 🔸 `st.pydeck_chart()`

- **Used With**: PyDeck for WebGL 3D maps

### ✅ Syntax:
```python
st.pydeck_chart(deck_chart, use_container_width=False)
```

---

## 🔸 `st.vega_lite_chart()`

- **Used With**: Raw Vega-Lite specifications

### ✅ Syntax:
```python
st.vega_lite_chart(data, spec, use_container_width=False)
```

---

## 🧩 Example Usage

```python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Matplotlib
fig, ax = plt.subplots()
ax.plot(np.random.randn(50))
st.pyplot(fig)

# Plotly
df = px.data.iris()
fig2 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig2)

# Altair
import altair as alt
chart = alt.Chart(df).mark_circle().encode(
    x='sepalLength', y='sepalWidth', color='species'
)
st.altair_chart(chart)
```

---

## 📌 Summary of Use-Cases

| Goal                              | Recommended Library |
|-----------------------------------|---------------------|
| Static plots                      | Matplotlib          |
| Statistical visualization         | Seaborn             |
| Interactivity with zoom, hover    | Plotly              |
| Declarative and layered charts    | Altair              |
| Interactive dashboards (JS-based) | Bokeh               |
| Geospatial 3D data                | PyDeck              |
| Spec-driven design                | Vega-Lite           |

---
