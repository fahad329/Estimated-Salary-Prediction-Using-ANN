import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

st.title('Estimated Salary Prediction')

#loading the train model
model = tf.keras.models.load_model('model_reg.h5')

#loading the encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#userinputs

geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])
customerchurn = st.selectbox('Is customer churn',[0,1])

#preparing the data

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'CustomerChurn':[customerchurn]

})

#encoding geography column
encoder_geo = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(encoder_geo,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#concatinating encoded and input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


prediction = model.predict(input_data)
prediction_proba = prediction[0][0]

st.write('Estimated Salary is Approximately: ',prediction_proba)