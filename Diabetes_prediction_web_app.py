# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 01:00:02 2026

@author: nitin
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open("trained_model.sav",'rb'))

#creating a function for prediction 

def diabetes_prediction(input_data):
    input_data = (10,139,80,0,0,27.1,1.441,57)



    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
      return "The person is not diabetic"
    else:
      return "The person is diabetic"
  
def main():
    
    #giving title
    st.title("Diabetes Prediction App")
    
    #getting input data from user 
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
  
    #code for prediction
    
    diagnosis=''
    
    #creating a button for prediction
    if st.button("Diabetes Test Results"):
        diagnosis= diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
  
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
  
    
      
