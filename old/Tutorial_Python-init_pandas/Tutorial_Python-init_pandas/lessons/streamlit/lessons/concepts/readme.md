## Core Concepts
- **App Structure**
  - Script-based execution
  - Caching for performance
  - Session state management
- **Widgets**
  - Buttons, checkboxes, sliders
  - Text inputs, number inputs, file uploader
  - Date pickers, multi-selects, dropdowns
- **Layouts & Containers**
  - Sidebar, columns, expandable sections
  - Tabs, empty spaces, dividers
- **Displaying Data**
  - Text elements (title, header, markdown)
  - Tables, dataframes
  - Metric display, JSON output
- **Media Handling**
  - Images, videos, audio
- **Charts & Visualization**
  - Matplotlib, Seaborn, Plotly
  - Altair, Bokeh, Vega-Lite
- **Data Interaction**
  - Forms and input collection
  - File handling (CSV, Excel, JSON)
- **Session State**
  - User interaction memory
  - Callback functions
- **Advanced Features**
  - Custom components
  - Theming and configuration
  - URL parameters handling
- **Deployment**
  - Local execution
  - Streamlit Cloud, Docker, AWS, GCP


---
---


## **concepts and sub-concepts** in Streamlit:

---

### **1. Installation and Setup**
- Installing Streamlit
- Running Streamlit Apps (`streamlit run <app_name>.py`)
- Updating Streamlit
- Configuration Options (`config.toml`)

---

### **2. Basic Components**
#### **a. Text Display**
- `st.title()`
- `st.header()`
- `st.subheader()`
- `st.markdown()`
- `st.caption()`
- `st.code()`
- `st.latex()`

#### **b. Data Display**
- `st.write()`
- `st.json()`
- `st.dataframe()`
- `st.table()`
- `st.metric()`

#### **c. Media**
- `st.image()`
- `st.audio()`
- `st.video()`

#### **d. User Input Widgets**
- Text Input:
  - `st.text_input()`
  - `st.text_area()`
- Numeric Input:
  - `st.number_input()`
- Date/Time Input:
  - `st.date_input()`
  - `st.time_input()`
- Selections:
  - `st.selectbox()`
  - `st.multiselect()`
  - `st.radio()`
  - `st.checkbox()`
- Sliders:
  - `st.slider()`
  - `st.select_slider()`
- File Upload:
  - `st.file_uploader()`
- Buttons:
  - `st.button()`
  - `st.form_submit_button()`
- Others:
  - `st.color_picker()`

---

### **3. Layout and Structure**
#### **a. Containers**
- `st.container()`
- `st.expander()`
- `st.columns()`

#### **b. Sidebars**
- `st.sidebar.<element>()`

#### **c. Tabs**
- `st.tabs()`

#### **d. Layout Management**
- `st.set_page_config()`
  - Parameters: `page_title`, `page_icon`, `layout`, `initial_sidebar_state`

---

### **4. Charts and Visualization**
#### **a. Built-in Charting**
- `st.line_chart()`
- `st.area_chart()`
- `st.bar_chart()`
- `st.map()`

#### **b. Custom Charts**
- `st.pyplot()`
- `st.altair_chart()`
- `st.vega_lite_chart()`
- `st.plotly_chart()`
- `st.bokeh_chart()`
- `st.pydeck_chart()`
- `st.graphviz_chart()`

---

### **5. Interactivity and State Management**
#### **a. Session State**
- `st.session_state`
- Key-value storage and callbacks

#### **b. Forms**
- `st.form()`
- `st.form_submit_button()`

---

### **6. Theming**
- Customizing themes via `config.toml`
  - Colors, fonts, and styles

---

### **7. Advanced Features**
#### **a. Custom Components**
- Using external JavaScript or iframe-based components
- `streamlit-component-template`

#### **b. Caching**
- `@st.cache_data`
- `@st.cache_resource`

#### **c. Progress and Status**
- `st.progress()`
- `st.spinner()`
- `st.balloons()`
- `st.toast()` (Experimental)

#### **d. Error Handling**
- `st.exception()`
- `st.error()`
- `st.warning()`
- `st.success()`
- `st.info()`

#### **e. File Handling**
- `st.download_button()`

---

### **8. Deployment and Sharing**
- Streamlit Cloud
- Deploying on AWS/GCP/Heroku
- Sharing via Public URL

---

### **9. Debugging and Monitoring**
- Debug Mode (`streamlit run --global.logLevel debug`)
- `st.echo()`

---
