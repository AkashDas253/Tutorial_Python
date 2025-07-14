## **Basics of Streamlit**

### **1. What is Streamlit?**
- Streamlit is an open-source Python library used for building data-driven web applications quickly and easily.
- Ideal for machine learning engineers, data scientists, and developers to create interactive dashboards and prototypes.

---

### **2. Features of Streamlit**
- **Simple and Intuitive API**: Write Python scripts to create web apps without any web development knowledge.
- **Interactive Widgets**: Add buttons, sliders, and other widgets for user interaction.
- **Seamless Integration**: Integrates with popular Python libraries like NumPy, Pandas, Matplotlib, and Scikit-learn.
- **Live Code Updates**: Automatically refreshes the app when the code is saved.
- **Responsive Design**: Automatically adjusts the layout for different devices.

---

### **3. Installing Streamlit**
Streamlit can be installed using pip:
```bash
pip install streamlit
```

---

### **4. Running a Streamlit App**
1. Create a Python script (`app.py`):
   ```python
   import streamlit as st

   st.title("Hello, Streamlit!")
   st.write("This is your first Streamlit app.")
   ```
2. Run the script in the terminal:
   ```bash
   streamlit run app.py
   ```
3. The app will open in the browser at `http://localhost:8501`.

---

### **5. Streamlit Command Line Options**
- **Run a Streamlit App**:
  ```bash
  streamlit run <script_name.py>
  ```
- **Clear Cache**:
  ```bash
  streamlit cache clear
  ```
- **View Version**:
  ```bash
  streamlit version
  ```
- **Generate Config File**:
  ```bash
  streamlit config show
  ```

---

### **6. Key Streamlit Functions**
#### **a. Basic Display Elements**
- `st.title()`: Displays the app title.
- `st.header()`, `st.subheader()`: Add headings.
- `st.text()`: Displays plain text.
- `st.markdown()`: Displays formatted text using Markdown.
- `st.write()`: A versatile function for displaying text, data, or plots.

#### **b. Example Code**
```python
import streamlit as st

st.title("Streamlit Basics")
st.header("Introduction to Streamlit")
st.text("Streamlit makes it easy to build interactive web apps.")
```

---

### **7. Anatomy of a Streamlit App**
A Streamlit app is built by combining:
- **Python Script**: The logic of the app.
- **Streamlit Functions**: Components for UI and interaction.
- **Data Processing**: Libraries like Pandas or NumPy for computation.

#### Example App:
```python
import streamlit as st
import pandas as pd

st.title("Basic Streamlit App")
st.write("Streamlit simplifies web app development.")

# Add a slider
age = st.slider("Select your age", 0, 100)

# Display selected value
st.write(f"You selected: {age}")

# Data display
data = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
st.dataframe(data)
```

---

### **8. Key Points to Note**
- **Stateless Nature**: Streamlit apps reset on every rerun, ensuring consistency but requiring state management for complex interactions.
- **Script-based**: Streamlit follows a script execution model where code is executed from top to bottom.

---

### **9. Next Steps**
- Learn about **User Interface Components** to add interactivity.
- Explore **Layouts and Containers** for organizing your app.

---

