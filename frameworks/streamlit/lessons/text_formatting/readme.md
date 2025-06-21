## **Text and Formatting in Streamlit**

Streamlit provides various functions to display and format text. These allow you to create visually appealing and well-structured applications.

---

### **1. Basic Text Functions**

#### **a. `st.title()`**
- Displays a large title at the top of the app.
- Ideal for the app name or main heading.
- **Example**:
  ```python
  st.title("Welcome to Streamlit!")
  ```

#### **b. `st.header()`**
- Displays a smaller heading.
- Used to introduce sections in the app.
- **Example**:
  ```python
  st.header("Section 1: Introduction")
  ```

#### **c. `st.subheader()`**
- Displays a subheading.
- Used for subsections under a header.
- **Example**:
  ```python
  st.subheader("Subsection 1.1: Overview")
  ```

#### **d. `st.text()`**
- Displays plain text without formatting.
- Useful for adding simple descriptions or instructions.
- **Example**:
  ```python
  st.text("This is plain text displayed in Streamlit.")
  ```

#### **e. `st.write()`**
- A versatile function that displays various data types like text, Pandas DataFrames, or plots.
- Automatically infers the best way to render the content.
- **Example**:
  ```python
  st.write("This is a **formatted text** with Markdown.")
  st.write({"key": "value", "number": 42})
  ```

---

### **2. Text Formatting with Markdown**

Streamlit supports Markdown formatting for enhanced text presentation.

#### **a. Bold and Italics**
- **Bold**: `**text**`
- *Italic*: `*text*`
- **Example**:
  ```python
  st.markdown("This is **bold text** and *italic text*.")
  ```

#### **b. Lists**
- **Unordered List**:
  - Use `-` or `*` for list items.
  - **Example**:
    ```python
    st.markdown("- Item 1\n- Item 2\n- Item 3")
    ```
- **Ordered List**:
  - Use numbers followed by a period.
  - **Example**:
    ```python
    st.markdown("1. First Item\n2. Second Item\n3. Third Item")
    ```

#### **c. Links**
- Use `[text](URL)` to create a hyperlink.
- **Example**:
  ```python
  st.markdown("[Visit Streamlit](https://streamlit.io)")
  ```

#### **d. Code Blocks**
- Inline Code: Enclose code in backticks: `` `code` ``
- Code Block: Use triple backticks (` ``` `) for multi-line code.
- **Example**:
  ```python
  st.markdown("This is inline code: `print('Hello')`")
  st.markdown("""
  ```
  def hello():
      print("Hello, World!")
  ```
  """)
  ```

#### **e. Quotes**
- Use `>` for block quotes.
- **Example**:
  ```python
  st.markdown("> This is a block quote.")
  ```

---

### **3. Advanced Text Rendering**

#### **a. `st.markdown()`**
- Supports Markdown syntax and customization.
- **Example**:
  ```python
  st.markdown("### Markdown Heading")
  st.markdown("Here is some **bold text** with *italic text*.")
  ```

#### **b. `st.latex()`**
- Renders LaTeX for mathematical expressions.
- **Example**:
  ```python
  st.latex(r"E = mc^2")
  ```

#### **c. Custom HTML and CSS**
- Streamlit does not directly support HTML/CSS, but you can inject them using `st.markdown()` with `unsafe_allow_html=True`.
- **Example**:
  ```python
  st.markdown("<h1 style='color:blue;'>Blue Title</h1>", unsafe_allow_html=True)
  ```

---

### **4. Examples of Combined Formatting**

#### Example 1: Simple Text
```python
import streamlit as st

st.title("Streamlit Text and Formatting")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is plain text.")
```

#### Example 2: Markdown Formatting
```python
st.markdown("## Markdown Example")
st.markdown("""
- **Bold Item**
- *Italic Item*
- [Streamlit Link](https://streamlit.io)
""")
```

#### Example 3: LaTeX and Code
```python
st.subheader("Mathematical Formula")
st.latex(r"\int_a^b f(x) dx")

st.subheader("Code Example")
st.markdown("""
```python
def greet():
    return "Hello, Streamlit!"
```
""")
```

---
