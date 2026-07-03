import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent#Load the trained model
model=tf.keras.models.load_model(BASE_DIR / 'model.h5')

#Load the OneHotEncoder and LabelEncoder objects
with open(BASE_DIR / 'ohe_geography.pkl', 'rb') as f:
    ohe_geography = pickle.load(f)

with open(BASE_DIR / 'le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

#Load the StandardScaler object
with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

##Streamlitapp
st.title('Customer Churn Prediction')

#User_input
st.header('Enter Customer Details:')
geography = st.selectbox('Geography', ohe_geography.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'geo_encoded = ohe_geography.transform([[geography]]).toarray()
geo_encoded = ohe_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geography.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
