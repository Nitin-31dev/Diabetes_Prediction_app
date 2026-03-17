# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model=pickle.load(open("C:/Users/nitin/Documents/Data Science/Diabetes prediction project/trained_model.sav",'rb'))
# input data
input_data = (10,139,80,0,0,27.1,1.441,57)



input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("The person is not diabetic")
else:
  print("The person is diabetic")
