## **Charts**

---

#### **Built-in Charting**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.line_chart()`               | Renders a line chart based on input data.                                                             | `data (pandas.DataFrame, numpy.ndarray, or list)`<br> Default: None                                                                                 |
| `st.area_chart()`               | Renders an area chart based on input data.                                                            | `data (pandas.DataFrame, numpy.ndarray, or list)`<br> Default: None                                                                                 |
| `st.bar_chart()`                | Renders a bar chart based on input data.                                                              | `data (pandas.DataFrame, numpy.ndarray, or list)`<br> Default: None                                                                                 |
| `st.map()`                      | Renders a map with locations specified in the input data.                                            | `data (pandas.DataFrame)`<br> Default: None<br> `zoom (int)`<br> Default: 10<br> `use_container_width (bool)`<br> Default: False                       |

Example:
```python
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(100, 2), columns=["A", "B"])
st.line_chart(df)
```

---

#### **Custom Charts**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.pyplot()`                   | Renders a Matplotlib chart.                                                                          | `fig (matplotlib.figure.Figure)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                                              |
| `st.altair_chart()`             | Renders an Altair chart.                                                                              | `chart (alt.Chart)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                                                          |
| `st.vega_lite_chart()`          | Renders a Vega-Lite chart.                                                                            | `chart (vega_lite.vl.View)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                                                 |
| `st.plotly_chart()`             | Renders a Plotly chart.                                                                              | `figure_or_data (plotly.graph_objects.Figure or dict)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                       |
| `st.bokeh_chart()`              | Renders a Bokeh chart.                                                                               | `figure (bokeh.plotting.figure.Figure)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                                      |
| `st.pydeck_chart()`             | Renders a Deck.gl chart for geospatial visualization.                                                 | `deck (pydeck.Deck)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                                                        |
| `st.graphviz_chart()`           | Renders a Graphviz graph.                                                                            | `dot_source (str)`<br> Default: None<br> `use_container_width (bool)`<br> Default: False                                                          |

---
