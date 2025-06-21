## **Control Flow in Streamlit**

Control flow in Streamlit refers to the way the app reacts to user interactions and dynamically updates based on inputs. It is achieved through conditional statements, loops, and Streamlit-specific state management.

---

### **Key Components of Control Flow**

#### **1. Conditional Statements**
- Used to display content based on user input or other conditions.

**Example:**
```python
option = st.selectbox("Choose an option:", ["Option A", "Option B"])
if option == "Option A":
    st.write("You selected Option A")
else:
    st.write("You selected Option B")
```

---

#### **2. Loops**
- Useful for dynamically generating content.

**Example:**
```python
for i in range(5):
    st.write(f"This is item {i+1}")
```

---

#### **3. Callback Functions**
- Execute specific actions when an event occurs, such as clicking a button.

**Example:**
```python
def on_button_click():
    st.write("Button clicked!")

st.button("Click me", on_click=on_button_click)
```

---

### **State Management**

Streamlit uses `st.session_state` for managing variables that persist across reruns.

#### **4. Storing and Accessing State Variables**
- Use `st.session_state` to store and modify state variables.

**Example:**
```python
if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1

st.write(f"Count: {st.session_state.count}")
```

---

#### **5. Resetting State Variables**
- Reset the state to an initial value.

**Example:**
```python
if st.button("Reset"):
    st.session_state.count = 0
st.write(f"Count: {st.session_state.count}")
```

---

### **6. Progress Bars**
- Control flow often includes displaying progress.

**Example:**
```python
import time
progress = st.progress(0)
for i in range(101):
    time.sleep(0.05)
    progress.progress(i)
st.write("Task completed!")
```

---

### **7. Dynamic Content Updates**
- Placeholders (`st.empty()`) allow you to dynamically update content.

**Example:**
```python
placeholder = st.empty()
for i in range(5):
    placeholder.write(f"Updating... {i+1}")
    time.sleep(1)
placeholder.write("Done!")
```

---

### **8. Error Handling**
- Use `try-except` blocks to manage errors gracefully.

**Example:**
```python
try:
    value = st.number_input("Enter a number:")
    result = 100 / value
    st.write(f"Result: {result}")
except ZeroDivisionError:
    st.error("Division by zero is not allowed!")
```

---

### **9. Control Flow with Interactive Widgets**
Widgets dynamically affect control flow by rerunning the script.

**Example:**
```python
if st.checkbox("Show/Hide Details"):
    st.write("Here are the details!")
else:
    st.write("Details hidden.")
```

---

### **10. Break and Continue in Loops**
- Use `break` and `continue` in loops to modify flow.

**Example:**
```python
for i in range(10):
    if i == 5:
        break  # Exit loop
    if i % 2 == 0:
        continue  # Skip even numbers
    st.write(f"Odd number: {i}")
```

---

### Best Practices
1. **Minimize Re-Runs**: Use `st.session_state` to manage stateful logic efficiently.
2. **Separate Logic**: Keep UI and business logic separate for clarity.
3. **Dynamic Updates**: Use placeholders (`st.empty()`) and callbacks to refresh specific sections.

---
