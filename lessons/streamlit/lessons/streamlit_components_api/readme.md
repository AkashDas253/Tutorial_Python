# Streamlit Components API

The **Streamlit Components API** allows developers to integrate custom frontend elements into their Streamlit apps using JavaScript, HTML, and CSS. This opens up a wide range of possibilities for building interactive, highly customized applications. Components can be used to add unique widgets, complex visualizations, or integrate third-party libraries that Streamlit does not natively support.

---

## What is the Streamlit Components API?

Streamlit Components are built using the **`streamlit.components.v1`** module. This API allows the creation of custom interactive widgets that are rendered on the frontend using standard web technologies (JavaScript, HTML, CSS).

| Feature                  | Description                                         |
|--------------------------|-----------------------------------------------------|
| **Custom UI**             | Use JavaScript and React to create custom UIs       |
| **Two-way Communication** | Exchange data between Streamlit Python and JS code  |
| **Reusability**           | Components can be reused across multiple Streamlit apps |
| **Integration with JS**   | Add rich interactivity, visualizations, or third-party libraries |

---

## Core Concepts

### 1. **Components Declaration (`declare_component`)**
- Define a custom component by declaring it through `declare_component` method.
- **Backend (Python)** communicates with **Frontend (JavaScript)** by passing parameters.

**Syntax Example:**
```python
import streamlit.components.v1 as components

# Declare a component and pass props
component = components.declare_component("my_component", path="frontend")
```

### 2. **Frontend Implementation (HTML/JS)**

- **Frontend**: You write HTML, CSS, and JavaScript using frameworks like React, Vue, or Vanilla JS.
- **Backend**: Python interacts with the frontend via function calls.

---

## Creating and Using Components

### 1. **Basic Usage**

- **Frontend (JavaScript)** code is hosted separately or directly inside the Streamlit app's folder.
- Custom components can be integrated into Python-based workflows seamlessly.

**Example - Using `st.text_input` as a custom component:**
```python
import streamlit.components.v1 as components

# Create a simple custom input widget in HTML
html_code = """
<input type="text" id="input1" placeholder="Enter something">
"""
components.html(html_code, height=100)
```

---

### 2. **Passing Data to Components**

You can pass Python data to the custom components using `args` (arguments) or `props`.

**Python-to-JS Communication Example:**
```python
import streamlit as st
import streamlit.components.v1 as components

# Pass data to the component
html_code = """
<p>The number passed is: {{ value }}</p>
"""
components.html(html_code, height=200, args={"value": 42})
```

---

### 3. **Returning Data from Components**

- Components can return data back to Python using `component_value` or event handlers (e.g., button clicks).
- **Use cases** include forms, game boards, or drawing apps.

**Python-to-JS and JS-to-Python Example:**
```python
import streamlit.components.v1 as components

def my_component():
    return components.declare_component("my_component", path="frontend")

# Retrieve value from component
value_from_component = my_component()
```

---

## Example: Custom JavaScript Component

Here’s how you can create an interactive component that communicates between Python and JavaScript:

### Frontend (React/JS)
```javascript
const { Streamlit } = window;
Streamlit.setComponentValue("component_id", "clicked");

// Handle incoming props and events from Python
Streamlit.onReady(function () {
  const button = document.getElementById("myButton");
  button.addEventListener("click", function () {
    Streamlit.setComponentValue("clicked");
  });
});
```

### Backend (Python)
```python
import streamlit.components.v1 as components
clicked = components.declare_component("button_component", path="frontend")

if clicked:
    st.write("Button was clicked!")
```

---

## Customization and Advanced Features

### 1. **Embedding Third-Party Libraries**

- **Libraries like Plotly, D3.js**, and others can be directly embedded into Streamlit apps through custom components, giving you access to their full capabilities.
  
Example: Embedding a custom Plotly chart:
```python
import streamlit.components.v1 as components

html_code = """
<script>
  var data = [{x: [1, 2, 3], y: [10, 11, 12]}];
  Plotly.newPlot('plotly-plot', data);
</script>
"""
components.html(html_code, height=400)
```

---

### 2. **Styling and Design**

Custom components allow full flexibility over the design and styling of elements in Streamlit apps. This includes the use of **CSS**, **JS libraries**, and custom animations.

---

## Performance Considerations

| Feature                   | Description |
|---------------------------|-------------|
| **Optimized rendering**    | Use minimal re-renders and only update components when needed |
| **Async support**          | Use async calls for long-running JS tasks |
| **State management**       | Efficiently manage state to avoid unnecessary recomputation |

---

## Example: Creating a Full Custom Component

A complete custom component could integrate a dynamic chart library with a form, allowing users to interact with a graph in real-time.

**Python + JavaScript Integration:**
- `st.text_input` or `st.slider` allows user input in the app.
- Use JavaScript and a chart library like **Chart.js** or **Plotly.js** to update the visualization in real-time.

```python
import streamlit as st
import streamlit.components.v1 as components

# Create slider
value = st.slider("Select value", min_value=1, max_value=100)

# Declare a custom component for interactive chart
chart = components.declare_component("interactive_chart", path="frontend")

# Use slider value to update chart dynamically
chart(value=value)
```

---

## Advanced Integration Techniques

### 1. **Two-Way Communication**
Components allow **bidirectional communication** between Python and JavaScript. For example, custom input components can send data back to Python, and Python can change the state of the frontend.

### 2. **Using Web APIs**
Streamlit components can integrate **REST APIs** or **WebSocket connections** to communicate with external services, updating your app dynamically with external data.

---

## Use Cases of Streamlit Components API

- **Advanced Visualization**: Integrating libraries such as Plotly, Three.js, and D3 for highly customized charts.
- **Custom Widgets**: Building input widgets (e.g., sliders, date pickers) tailored to specific needs.
- **Interactive Maps**: Embedding interactive maps (using Leaflet.js or Google Maps).
- **Complex Forms**: Create multi-step forms with custom validation.
- **Real-Time Data**: Streaming data visualizations with WebSocket integration.

---

## Conclusion

The **Streamlit Components API** gives developers full control over the app’s frontend. It allows for creating **rich, interactive UIs** with complex behavior, incorporating advanced libraries, or even building entirely new widgets. By seamlessly combining Python’s simplicity with JavaScript’s flexibility, the API extends Streamlit’s capabilities far beyond its basic components.
