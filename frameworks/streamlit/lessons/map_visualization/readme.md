# Comprehensive Note on **Map Visualization** in Streamlit

Streamlit supports geospatial data visualization using simple built-in methods as well as integration with advanced libraries like **Pydeck**, **Plotly**, and **Folium**.

---

## Built-in Map Function: `st.map()`

- Quick and simple way to plot points on a map using latitude and longitude.

### Syntax:
```python
st.map(data, zoom=10, use_container_width=True)
```

| Parameter             | Description                                         |
|-----------------------|-----------------------------------------------------|
| `data`                | DataFrame with `lat` and `lon` columns              |
| `zoom`                | Initial zoom level (default is `10`)                |
| `use_container_width`| Adjusts map width to the layout (default: `True`)   |

### Example:
```python
import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'lat': [37.76, 37.77, 37.78],
    'lon': [-122.4, -122.41, -122.42]
})

st.map(df)
```

---

## Advanced Map: `st.pydeck_chart()`

- Enables rich geospatial 2D/3D maps using **PyDeck** (WebGL-powered).

### Syntax:
```python
st.pydeck_chart(pydeck_obj, use_container_width=False)
```

### Example with Pydeck:
```python
import pydeck as pdk

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    get_radius=200,
)

view_state = pdk.ViewState(
    latitude=37.77,
    longitude=-122.41,
    zoom=12,
    pitch=50,
)

chart = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state
)

st.pydeck_chart(chart)
```

---

## Using Plotly for Maps

- **Plotly Express** supports scatter maps and choropleths.

```python
import plotly.express as px

fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=10)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)
```

---

## Optional: Folium via Components

- For more custom maps (heatmaps, tile layers), Folium can be used via `streamlit_folium`.

```bash
pip install streamlit-folium
```

```python
from streamlit_folium import folium_static
import folium

m = folium.Map(location=[37.77, -122.41], zoom_start=13)
folium.Marker([37.76, -122.4], tooltip="Point A").add_to(m)

folium_static(m)
```

---

## Summary Table

| Method              | Library      | Best For                                | Interactivity |
|---------------------|--------------|------------------------------------------|---------------|
| `st.map()`          | Built-in     | Simple geo-point plotting                | ❌ Basic only |
| `st.pydeck_chart()` | PyDeck       | WebGL 2D/3D visualization                | High       |
| `st.plotly_chart()` | Plotly       | Interactive map-based scatter plots      | High       |
| `folium_static()`   | Folium       | Custom Leaflet-based maps                | Moderate   |

---
