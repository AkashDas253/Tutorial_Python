## **Layout and Structure**

---

#### **Containers**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.container()`                | Creates a container to organize widgets and components.                                              | No parameters                                                                                                                                      |
| `st.expander()`                 | Creates an expandable container to hide or show content based on user interaction.                   | `label (str)`<br> Default: None<br> `expanded (bool)`<br> Default: `False`                                                                        |
| `st.columns()`                  | Creates multiple columns to organize content side by side.                                           | `n_columns (int)`<br> Default: 1<br> `gap (int)`<br> Default: 40                                                                                 |

Example:
```python
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")
```

---

#### **Sidebars**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.sidebar.<element>()`        | Used for rendering elements in the sidebar (can be used with many widgets like `st.selectbox()`, etc.). | Parameters depend on the specific element used in the sidebar, e.g., `st.sidebar.selectbox()`.                                                   |

Example:
```python
st.sidebar.selectbox("Choose an option", options=["Option 1", "Option 2"])
```

---

#### **Tabs**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.tabs()`                     | Creates tabbed navigation to organize content into multiple sections.                                | `labels (list)`<br> Default: `[]`                                                                                                                 |

Example:
```python
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
tab1.write("Content for Tab 1")
tab2.write("Content for Tab 2")
```

---

#### **Layout Management**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.set_page_config()`          | Allows customization of the page layout and sidebar behavior.                                        | `page_title (str)`<br> Default: None<br> `page_icon (str or PIL.Image)`<br> Default: None<br> `layout (str)`<br> Default: "centered"<br> `initial_sidebar_state (str)`<br> Default: "auto" |

Example:
```python
st.set_page_config(page_title="My Streamlit App", page_icon=":guardsman:", layout="wide")
```

---
