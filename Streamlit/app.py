import pandas as pd
import streamlit as st
import numpy as np

st.title("Hello Streamlit")

st.write('This is a simple Streamlit app to demonstrate how to use Streamlit for building interactive web applications. Below is an example of a DataFrame displayed using Streamlit.')

df=pd.DataFrame({
    'Column 1': [1, 2, 3, 4, 5],
    'Column 2': ['A', 'B', 'C', 'D', 'E']
})
st.write('Here is a sample DataFrame:')
st.dataframe(df)

chart_data=pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)