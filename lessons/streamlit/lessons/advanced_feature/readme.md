## **Advanced Features in Streamlit**

---

### **Custom Components**

Streamlit allows you to create custom components by embedding external JavaScript or HTML. This can be achieved through `streamlit.components.v1`.

- **Using JavaScript or HTML**: You can use this to integrate custom HTML or JavaScript components into your Streamlit app.
  
  Example:
  ```python
  import streamlit as st
  import streamlit.components.v1 as components
  components.html("<h1>Custom HTML with JavaScript</h1>")
  ```

- **`streamlit-component-template`**: A starter template to help you build and integrate custom components. It simplifies the process of creating custom components using popular frontend frameworks like React or Vue.

---

### **Caching**

Streamlit offers caching to store the results of expensive operations like data loading or model predictions to improve performance.

| **Caching Decorators**         | **Description**                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| `@st.cache_data`                | Caches the output of a function based on input data. It is re-executed only if the input changes.    |
| `@st.cache_resource`            | Caches larger external resources such as models or datasets to prevent reloading them every time.    |

Example:
```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

---

### **Progress and Status**

Streamlit provides tools to show progress, loading indicators, and status updates to users during long-running tasks.

| **Progress and Status Widgets** | **Description**                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| `st.progress()`                 | Displays a progress bar that tracks task completion.                                                  |
| `st.spinner()`                  | Displays a spinning indicator to show that a task is in progress.                                    |
| `st.balloons()`                 | Displays balloons as a success indicator after a task completes.                                      |
| `st.toast()` (Experimental)      | Shows a brief message (toast) at the top of the app after a task completes.                          |

Example:
```python
st.progress(50)
```

---

### **Error Handling**

Streamlit provides various functions to display error messages and other types of notifications, making it easier to handle different situations.

| **Error Handling Widgets**      | **Description**                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------------|
| `st.exception()`                 | Displays detailed information about an exception, including the stack trace.                        |
| `st.error()`                     | Displays an error message in red, typically for errors in execution.                                 |
| `st.warning()`                   | Displays a warning message in yellow, used for potential issues that need attention.                |
| `st.success()`                   | Displays a success message in green, typically indicating successful operations.                     |
| `st.info()`                      | Displays informational messages in blue, providing additional details or guidance.                   |

Example:
```python
st.error("This is an error message.")
```

---

### **File Handling**

Streamlit provides a straightforward method for uploading and downloading files within your app.

| **File Handling Widgets**        | **Description**                                                                                      |
|-----------------------------------|------------------------------------------------------------------------------------------------------|
| `st.download_button()`            | Allows users to download files or data directly from the app.                                        |

Example:
```python
st.download_button("Download CSV", data, "file.csv")
```

---

### **Dynamic Updates**

Streamlit enables dynamic updates, allowing your app to refresh or re-run in response to user interactions.

- **`st.experimental_rerun()`**: Forces the app to rerun, allowing for dynamic updates when necessary.
  
  Example:
  ```python
  st.experimental_rerun()
  ```

- **`st.session_state`**: This is used to store values that persist across app reruns, ensuring state management between different interactions.
  
  Example:
  ```python
  if "counter" not in st.session_state:
      st.session_state.counter = 0
  ```

---

### **Themes and Customization**

Streamlit allows for customization of the app’s appearance by modifying themes and other visual elements.

- **Custom Themes**: You can change colors, fonts, and other design elements to match your app’s branding.

  Example (in `config.toml`):
  ```toml
  [theme]
  primaryColor = "#FF5733"
  backgroundColor = "#FFFFFF"
  ```

---
