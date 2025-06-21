## **Complete Streamlit Cheatsheet** 

---

## Initialization

```bash
pip install streamlit
streamlit run your_app.py
```

---

## Display Elements

```python
st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.text("Plain text")
st.markdown("**Markdown** _text_")
st.code("print('Hello')", language='python')
st.latex(r"\alpha^2 + \beta^2 = \gamma^2")
```

---

## Status Messages

```python
st.success("Success!")
st.info("Info")
st.warning("Warning!")
st.error("Error")
st.exception(Exception("Test"))
```

---

## Data Display

```python
st.write(obj)                   # Auto-detects type
st.table(data)                  # Static table
st.dataframe(df, width=700)     # Interactive table
st.json(data)                   # Pretty JSON viewer
```

---

## Media Elements

```python
st.image("img.png", width=300)
st.video("video.mp4")
st.audio("audio.mp3")
```

---

## Input Widgets

```python
st.button("Click")
st.checkbox("Check")
st.radio("Pick one", ["A", "B"])
st.selectbox("Choose", options)
st.multiselect("Pick many", options)
st.slider("Slide", min_value, max_value, step)
st.text_input("Text")
st.text_area("Textarea")
st.number_input("Number")
st.date_input("Date")
st.time_input("Time")
st.file_uploader("Upload file")
st.color_picker("Pick color")
```

---

## Forms (Grouped Inputs)

```python
with st.form("my_form"):
    name = st.text_input("Name")
    submitted = st.form_submit_button("Submit")
```

---

## Layouts and Containers

```python
st.sidebar.title("Sidebar")
st.columns([1, 2])                 # Horizontal layout
with st.expander("See more"):     # Collapsible section
    st.write("Expanded content")

with st.container():              # Grouped content
    st.write("Inside container")
```

---

## Control Flow

```python
if st.button("Next"):
    do_something()
```

---

## Charts and Visualization

```python
st.line_chart(data)
st.area_chart(data)
st.bar_chart(data)
st.map(df)                             # Requires lat/lon columns
st.pyplot(fig)                         # Matplotlib
st.plotly_chart(fig)                   # Plotly
st.altair_chart(chart)                 # Altair
st.vega_lite_chart(data, spec)        # Vega
st.deck_gl_chart(data)                # Pydeck
```

---

## State Management

```python
if "count" not in st.session_state:
    st.session_state.count = 0

st.session_state.count += 1
st.write(st.session_state.count)
```

---

## Caching

```python
@st.cache_data
def load_data():
    return result

@st.cache_resource
def load_model():
    return model
```

---

## File I/O

```python
uploaded = st.file_uploader("Choose file")
st.download_button("Download", data="text", file_name="data.txt")
```

---

## Theming

`./.streamlit/config.toml`

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## Multi-page App

Place files inside the `/pages/` directory.

```
app.py
/pages/
  Page1.py
  Page2.py
```

---

## Run with Custom Parameters

```bash
streamlit run app.py -- --param1=abc
```

Use `sys.argv` or `argparse` in script.

---

## Useful CLI Commands

```bash
streamlit --help
streamlit config show
streamlit cache clear
streamlit docs
```

---

## Deployment

- Streamlit Community Cloud (1-click)
- Docker, Heroku, AWS, GCP

---
