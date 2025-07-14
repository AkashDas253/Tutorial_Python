## **Charts and Visualization in Streamlit**

Streamlit provides multiple methods for creating **interactive and static visualizations** using its built-in API as well as integration with popular libraries like **Matplotlib, Seaborn, Plotly, Altair, Pydeck**, and more.

---

## **Built-in Chart Functions**

| **Function**         | **Description**                           | **Input Format**         |
|----------------------|-------------------------------------------|--------------------------|
| `st.line_chart()`    | Line chart                                | Pandas DataFrame, Series |
| `st.area_chart()`    | Area chart                                | Pandas DataFrame, Series |
| `st.bar_chart()`     | Vertical bar chart                        | Pandas DataFrame, Series |
| `st.scatter_chart()` | Scatter plot (from version 1.30+)         | Pandas DataFrame         |
| `st.pyplot()`        | Display Matplotlib figure                 | `plt.figure()`           |
| `st.altair_chart()`  | Display Altair chart                      | Altair Chart object      |
| `st.plotly_chart()`  | Display Plotly figure                     | Plotly `go.Figure`       |
| `st.bokeh_chart()`   | Display Bokeh chart                       | Bokeh `Figure` object    |
| `st.vega_lite_chart()`| Quick Vega-Lite chart                    | Dict or Altair-like spec |
| `st.graphviz_chart()`| Render Graphviz diagrams                  | Dot language string      |
| `st.pydeck_chart()`  | Render geographic (WebGL) visualizations | Pydeck object            |

---

## **Built-in Chart Examples**

### **Line Chart**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.line_chart(df)
```

### **Bar Chart**
```python
st.bar_chart(df)
```

### **Area Chart**
```python
st.area_chart(df)
```

### **Scatter Chart**
```python
st.scatter_chart(df)
```

---

## **Integration with External Libraries**

### **Matplotlib**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
st.pyplot(fig)
```

### **Seaborn**
```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')
fig = sns.pairplot(df, hue='species')
st.pyplot(fig)
```

### **Plotly**
```python
import plotly.express as px

fig = px.scatter(df, x='a', y='b')
st.plotly_chart(fig)
```

### **Altair**
```python
import altair as alt

chart = alt.Chart(df).mark_line().encode(x='a', y='b')
st.altair_chart(chart, use_container_width=True)
```

### **Bokeh**
```python
from bokeh.plotting import figure

p = figure(title='Simple Line', x_axis_label='x', y_axis_label='y')
p.line([1, 2, 3], [4, 6, 2], line_width=2)
st.bokeh_chart(p)
```

---

## **Advanced Visualization**

### **Pydeck (Geospatial)**
```python
import pydeck as pdk

layer = pdk.Layer(
    'HexagonLayer',
    data=df,  # DataFrame with lat/lon
    get_position='[lon, lat]',
    radius=200,
    elevation_scale=4,
    elevation_range=[0, 1000],
    pickable=True,
    extruded=True,
)

view_state = pdk.ViewState(
    latitude=37.76,
    longitude=-122.4,
    zoom=11,
    pitch=50,
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
```

---

## **Graphviz for Diagrams**
```python
st.graphviz_chart('''
    digraph G {
        start -> process;
        process -> end;
    }
''')
```

---

## **Customizing Charts**

| **Customization**    | **Applies To**            | **Usage**                                         |
|----------------------|---------------------------|--------------------------------------------------|
| `use_container_width`| Altair, Plotly, Bokeh     | Fits chart to container width                    |
| Axis labels, titles  | All supported libraries   | Use library’s native customization API           |
| Interactivity        | Altair, Plotly, Pydeck    | Built-in by default                              |

---

## **Dynamic Data with Widgets**

Widgets like sliders, select boxes, or inputs can dynamically update charts.

```python
slider = st.slider("Number of rows", 10, 100, 20)
df = pd.DataFrame(np.random.randn(slider, 3), columns=['a', 'b', 'c'])
st.line_chart(df)
```

---

## **Best Practices**

- Prefer `st.line_chart`, `st.bar_chart`, and `st.area_chart` for fast prototyping.
- Use **Altair** or **Plotly** for interactivity and complex plots.
- Use **Pydeck** for geographical data.
- Cache data sources to improve performance using `@st.cache_data`.

---
