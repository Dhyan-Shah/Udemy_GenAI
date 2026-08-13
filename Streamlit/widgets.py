import streamlit as st
import pandas as pd
st.title("Streamlit Text Input")

name=st.text_input("Enter your name", "Type here...")
age=st.slider("Select your age", 0, 100, 25)
st.write(f"You are {age} years old.")


if name:
    st.write(f"Hello, {name}!")


options=['JavaScript', 'Python', 'Java', 'C++']
choice=st.selectbox("Select your favorite programming language", options)
st.write(f"You selected: {choice}") 

data=pd.DataFrame({
    'Column 1': [1, 2, 3, 4, 5],
    'Column 2': ['A', 'B', 'C', 'D', 'E']
})
st.write("Here is a sample DataFrame:")
st.dataframe(data)

uploaded_file=st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write("Here is the content of the uploaded file:")
    st.dataframe(df)