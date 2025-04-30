## **User Interface Components in Streamlit (With Parameters)**

Streamlit provides various user interface components to create interactive and responsive web apps. Below is a detailed table of these components with their parameters.

---

### **1. Text Display Components**

| **Function**    | **Description**                               | **Parameters**                                                                                                            |
|------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `st.title()`     | Displays a title.                            | `body: str` - Text of the title.                                                                                         |
| `st.header()`    | Displays a header.                           | `body: str` - Text of the header.                                                                                        |
| `st.subheader()` | Displays a subheader.                        | `body: str` - Text of the subheader.                                                                                     |
| `st.text()`      | Displays plain text.                         | `body: str` - Text to display.                                                                                           |
| `st.markdown()`  | Displays Markdown-formatted text.            | `body: str` - Markdown text.<br>`unsafe_allow_html: bool = False` - Allow raw HTML.                                       |
| `st.write()`     | Displays text, data, or plots dynamically.   | `*args` - Any data type to render (text, dict, DataFrame, etc.).                                                         |
| `st.latex()`     | Renders a mathematical formula using LaTeX.  | `body: str` - LaTeX string.                                                                                              |

---

### **2. Data Display Components**

| **Function**      | **Description**                           | **Parameters**                                                                                                  |
|--------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `st.dataframe()`   | Displays a DataFrame with interactions.  | `data: pd.DataFrame` - Data to display.<br>`width: int` - Width in pixels.<br>`height: int` - Height in pixels. |
| `st.table()`       | Displays a static table.                 | `data: pd.DataFrame or list` - Data to display.                                                                |
| `st.json()`        | Displays JSON or a dictionary.           | `body: dict` - JSON object to render.                                                                          |
| `st.metric()`      | Displays a metric widget.                | `label: str` - Metric label.<br>`value: str/int` - Current value.<br>`delta: str/int` - Change in value.        |

---

### **3. Input Widgets**

| **Function**      | **Description**                                        | **Parameters**                                                                                              |
|--------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `st.button()`      | Creates a button.                                      | `label: str` - Button text.<br>`key: str` - Unique key.<br>`on_click: callable` - Callback function.       |
| `st.checkbox()`    | Creates a checkbox.                                    | `label: str` - Checkbox label.<br>`value: bool` - Initial state.<br>`key: str` - Unique key.               |
| `st.radio()`       | Creates a radio button.                                | `label: str` - Radio label.<br>`options: list` - Options to select.<br>`key: str` - Unique key.            |
| `st.selectbox()`   | Creates a dropdown menu.                               | `label: str` - Dropdown label.<br>`options: list` - Options.<br>`key: str` - Unique key.                   |
| `st.multiselect()` | Creates a multi-select dropdown.                       | `label: str` - Dropdown label.<br>`options: list` - Options.<br>`default: list` - Default selections.      |
| `st.slider()`      | Creates a slider for numeric inputs.                   | `label: str` - Slider label.<br>`min_value, max_value` - Range.<br>`step: int` - Increment.                |
| `st.text_input()`  | Creates a text input box.                              | `label: str` - Input label.<br>`value: str` - Default value.<br>`key: str` - Unique key.                   |
| `st.text_area()`   | Creates a multi-line text input.                       | `label: str` - Input label.<br>`value: str` - Default value.<br>`height: int` - Box height.                |
| `st.number_input()`| Creates a numeric input box.                           | `label: str` - Input label.<br>`min_value, max_value` - Range.<br>`step: int/float` - Increment.           |
| `st.date_input()`  | Creates a date picker.                                 | `label: str` - Input label.<br>`value: datetime.date` - Default date.<br>`key: str` - Unique key.          |
| `st.file_uploader()`| Uploads files.                                        | `label: str` - Input label.<br>`type: str` - Allowed file types.<br>`accept_multiple_files: bool` - Allow multiple uploads. |

---

### **4. Media Display Components**

| **Function**       | **Description**                       | **Parameters**                                                                                                  |
|---------------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `st.image()`        | Displays images.                     | `image: str/array` - Image to display.<br>`caption: str` - Image caption.<br>`use_column_width: bool` - Fit width. |
| `st.audio()`        | Embeds audio files.                  | `data: file/array` - Audio source.<br>`format: str` - Audio format.<br>`start_time: int` - Start time in seconds. |
| `st.video()`        | Embeds video files.                  | `data: file` - Video source.<br>`format: str` - Video format.<br>`start_time: int` - Start time in seconds.     |

---

### **5. Layout and Containers**

| **Function**         | **Description**                                      | **Parameters**                                                                                     |
|-----------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `st.sidebar`          | Creates a sidebar for widgets and inputs.           | No parameters (use like a container).                                                            |
| `st.expander()`       | Creates an expandable section.                      | `label: str` - Expander title.<br>`expanded: bool` - Default state (open/closed).                |
| `st.columns()`        | Splits the layout into multiple columns.            | `n: int` - Number of columns.<br>`gap: str` - Spacing between columns (`small`, `medium`, `large`). |
| `st.container()`      | Groups elements into a container.                   | No parameters (use as a context manager).                                                        |
| `st.tabs()`           | Creates tabbed sections.                            | `labels: list` - Tab labels.                                                                     |

---

### **6. Example App with UI Components**

```python
import streamlit as st
import pandas as pd

st.title("Streamlit User Interface Components")

# Sidebar Example
st.sidebar.header("Sidebar Options")
option = st.sidebar.radio("Select an option", ["Option 1", "Option 2"])

# Text and Metrics
st.header("Metrics and Text")
st.metric(label="Temperature", value="30°C", delta="-2°C")

# Table and DataFrame
st.subheader("Data Table")
data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
st.table(df)
st.dataframe(df)

# Input Widgets
st.subheader("Inputs")
name = st.text_input("Enter your name")
age = st.slider("Select your age", 0, 100, 25)
file = st.file_uploader("Upload a file", type=["csv", "txt"])

# Display Results
if st.button("Submit"):
    st.write(f"Name: {name}, Age: {age}")
    if file:
        st.write("Uploaded File:", file.name)
```

--- 
