## **Layouts and Containers in Streamlit**

Streamlit provides powerful tools for arranging components on a page through layouts and containers. Below is a detailed explanation of the layouts, containers, and their use cases.

---

### **1. Columns**

| **Description** | Divides the screen into multiple columns. |
|------------------|-------------------------------------------|
| **Function**     | `st.columns()`                           |
| **Parameters**   |                                           |
| `spec`           | (int/list) Number of columns or relative sizes. |
| **Methods**      |                                           |
| `.write()`       | Write content in a specific column.      |

#### **Example:**
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.write("This is Column 1")
with col2:
    st.write("This is Column 2")
with col3:
    st.write("This is Column 3")
```

#### **Example with Relative Sizes:**
```python
col1, col2 = st.columns([2, 1])
col1.write("Column 1 is twice as wide as Column 2")
col2.write("Column 2")
```

---

### **2. Tabs**

| **Description** | Adds tabs for organizing content.             |
|------------------|----------------------------------------------|
| **Function**     | `st.tabs()`                                 |
| **Parameters**   |                                              |
| `labels`         | (list) List of tab labels.                  |
| **Methods**      |                                              |
| `.write()`       | Write content within a specific tab.        |

#### **Example:**
```python
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Content for Tab 1")
with tab2:
    st.write("Content for Tab 2")
```

---

### **3. Expander**

| **Description** | Creates a collapsible section for additional content. |
|------------------|-------------------------------------------------------|
| **Function**     | `st.expander()`                                       |
| **Parameters**   |                                                       |
| `label`          | (str) Text for the expander header.                   |
| **Methods**      |                                                       |
| `.write()`       | Write content within the expander.                    |

#### **Example:**
```python
with st.expander("More Information"):
    st.write("This is additional information inside an expander.")
```

---

### **4. Container**

| **Description** | A container that can hold multiple Streamlit elements. |
|------------------|--------------------------------------------------------|
| **Function**     | `st.container()`                                       |
| **Methods**      |                                                        |
| `.write()`       | Write content inside the container.                    |

#### **Example:**
```python
with st.container():
    st.write("This is inside the container")
    st.line_chart([1, 2, 3, 4])
```

---

### **5. Sidebar**

| **Description** | Adds a sidebar to the application.                     |
|------------------|--------------------------------------------------------|
| **Function**     | `st.sidebar`                                          |
| **Methods**      |                                                        |
| `.write()`       | Write content in the sidebar.                         |

#### **Example:**
```python
st.sidebar.title("Sidebar")
st.sidebar.write("This is the sidebar")
option = st.sidebar.selectbox("Choose an option", ["Option A", "Option B"])
st.write(f"You selected: {option}")
```

---

### **6. Empty**

| **Description** | Creates a placeholder for dynamically updating elements. |
|------------------|----------------------------------------------------------|
| **Function**     | `st.empty()`                                             |
| **Methods**      |                                                          |
| `.write()`       | Update the content in the placeholder.                  |

#### **Example:**
```python
placeholder = st.empty()
placeholder.write("This is a placeholder")

import time
for i in range(5):
    time.sleep(1)
    placeholder.write(f"Updating... {i+1}")
```

---

### **7. Horizontal Divider**

| **Description** | Adds a horizontal line for visual separation. |
|------------------|-----------------------------------------------|
| **Function**     | `st.divider()`                                |

#### **Example:**
```python
st.write("Above the divider")
st.divider()
st.write("Below the divider")
```

---

### **8. Pagination (Beta)**

| **Description** | Manages pages for organizing content (requires streamlit-multipage). |
|------------------|-----------------------------------------------------------------------|
| **Functionality** | Uses navigation between different Streamlit pages.                  |

#### **Example:**
```plaintext
Streamlit’s multipage feature is accessed via the "New Page" button in the Streamlit interface.
```

---

### **9. Vertical and Horizontal Spacers**

| **Description** | Adds empty space between elements. |
|------------------|------------------------------------|
| **Function**     | `st.write("")`                    |

#### **Example:**
```python
st.write("Above the space")
st.write("")  # Adds vertical space
st.write("Below the space")
```

---
