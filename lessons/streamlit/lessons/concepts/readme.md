## **Streamlit** concepts and their subconcepts:  

### 1. **Basics of Streamlit**
   - What is Streamlit
   - Installing Streamlit
   - Running a Streamlit App (`streamlit run app.py`)
   - Streamlit Command Line Options

### 2. **User Interface Components**
   - **Text and Formatting**
     - `st.title()`
     - `st.header()`
     - `st.subheader()`
     - `st.text()`
     - `st.markdown()`
     - `st.latex()`
     - `st.write()`
   - **Data Display**
     - `st.dataframe()`
     - `st.table()`
     - `st.json()`
     - `st.metric()`
   - **Media Elements**
     - `st.image()`
     - `st.audio()`
     - `st.video()`

### 3. **Input Widgets**
   - `st.button()`
   - `st.checkbox()`
   - `st.radio()`
   - `st.selectbox()`
   - `st.multiselect()`
   - `st.slider()`
   - `st.select_slider()`
   - `st.text_input()`
   - `st.text_area()`
   - `st.number_input()`
   - `st.date_input()`
   - `st.time_input()`
   - `st.file_uploader()`
   - `st.color_picker()`

### 4. **Layouts and Containers**
   - `st.sidebar`
   - `st.columns`
   - `st.expander()`
   - `st.container()`
   - `st.empty()`

### 5. **Control Flow**
   - `st.stop()`
   - `st.form()`
     - `st.form_submit_button()`
   - `st.spinner()`

### 6. **Charts and Visualization**
   - **Built-in Chart Functions**
     - `st.line_chart()`
     - `st.area_chart()`
     - `st.bar_chart()`
   - **Custom Visualizations**
     - `st.pyplot()`
     - `st.altair_chart()`
     - `st.plotly_chart()`
     - `st.bokeh_chart()`
     - `st.graphviz_chart()`
     - `st.vega_lite_chart()`
   - **Map Visualization**
     - `st.map()`

### 7. **State Management**
   - `st.session_state`
   - Callbacks and Widget Key Management

### 8. **Interactivity**
   - `st.experimental_data_editor()`
   - `st.experimental_get_query_params()`
   - `st.experimental_set_query_params()`

### 9. **Theming and Configurations**
   - Customizing themes with `.streamlit/config.toml`
   - Environment variables for configuration

### 10. **Advanced Features**
   - Caching with `@st.cache_data` and `@st.cache_resource`
   - File handling with `st.download_button()`
   - `st.experimental_memo` (deprecated)
   - Experimental components and APIs

### 11. **Deployment**
   - Deploying on **Streamlit Cloud**
   - Deploying on other platforms (Heroku, AWS, Azure, etc.)
   - Streamlit sharing options

### 12. **Extensions**
   - Streamlit Components API
   - Using Third-party Streamlit Components  
