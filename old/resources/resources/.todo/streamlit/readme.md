# Streamlit Cheatsheet

## 1. Installing Streamlit
- pip install streamlit  # Install Streamlit

## 2. Importing Libraries
- import streamlit as st  # Import Streamlit

## 3. Running Streamlit App
- streamlit run app.py  # Run the Streamlit app

## 4. Adding Title and Header
- st.title('My Streamlit App')  # Add a title
- st.header('Header Text')  # Add a header
- st.subheader('Subheader Text')  # Add a subheader

## 5. Adding Text
- st.write('This is some text!')  # Write text to the app
- st.markdown('# Markdown Text')  # Add markdown text

## 6. Adding Widgets
- st.button('Click Me')  # Add a button
- st.radio('Select an option:', ['Option 1', 'Option 2'])  # Add a radio button
- st.checkbox('Check Me')  # Add a checkbox
- st.selectbox('Select a number:', [1, 2, 3])  # Add a select box
- st.slider('Select a range:', 0, 100, 50)  # Add a slider

## 7. Uploading Files
- uploaded_file = st.file_uploader('Upload a file')  # File uploader
- if uploaded_file is not None:
  - # Process the uploaded file

## 8. Displaying DataFrames
- import pandas as pd  # Import Pandas
- df = pd.DataFrame({'Column 1': [1, 2], 'Column 2': [3, 4]})  # Create DataFrame
- st.dataframe(df)  # Display DataFrame

## 9. Plotting Charts
- import matplotlib.pyplot as plt  # Import Matplotlib
- fig, ax = plt.subplots()  # Create a figure
- ax.plot([1, 2, 3], [1, 4, 9])  # Plot data
- st.pyplot(fig)  # Display the plot

## 10. Using Caching
- @st.cache  # Cache the function
- def load_data():
  - # Load data function
- data = load_data()  # Call the cached function

## 11. Sidebar
- st.sidebar.title('Sidebar Title')  # Add a title to the sidebar
- st.sidebar.button('Sidebar Button')  # Add a button to the sidebar

## 12. Session State
- st.session_state['key'] = 'value'  # Set session state
- value = st.session_state.get('key')  # Get session state value
