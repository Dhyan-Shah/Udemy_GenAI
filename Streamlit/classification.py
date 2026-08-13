import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

st.title("Iris Classification with Streamlit")

@st.cache
def load_data():
    iris=load_iris()
    df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['Species']=iris.target
    return df, iris.target_names

df, target_names=load_data()

model=RandomForestClassifier()
model.fit(df.drop('Species', axis=1), df['Species'])

st.sidebar.title("Input Features")
sepal_length=st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width=st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length=st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width=st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

input_data=[[sepal_length, sepal_width, petal_length, petal_width]]
prediction=model.predict(input_data)
predicted_species=target_names[prediction][0]

st.write("### Input Features")
st.write(f"Sepal Length: {sepal_length} cm")
st.write(f"Sepal Width: {sepal_width} cm")
st.write(f"Petal Length: {petal_length} cm")
st.write(f"Petal Width: {petal_width} cm")

st.write(f"Predicted Iris Species: {predicted_species}")