from trainingModels import trainModel
from predictingModels.predictModel import predictModel1
from tempextraction import temperature
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
st.title("Solar power forecasting")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
html_temp = """
<div style ="background-color:lightblue;padding:13px">
<h1 style ="color:black;text-align:center;">Solar Power Generation Forecasting App </h1>
</div>
"""
    
# this line allows us to display the front end aspects we have 
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)
    
# the following lines create text boxes in which the user can enter 
# the data required to make the prediction
city = st.text_input("Enter the city name")

result =""
    
# the below line ensures that when the button called 'Predict' is clicked, 
# the prediction function defined above is called to make the prediction 
# and store it in the variable result

if st.button("Predict"):
    train_model = trainModel()
    temperature,pressure,humidity,zeinth,azimuth,bestmodel=train_model.trainingModels(city)
    predict_model=predictModel1()
    result=predict_model.predictionFromModel(temperature,pressure,humidity,zeinth,azimuth,bestmodel)
    st.info('The temperature of is:{}'.format(temperature))
    st.info('The pressure is :{}'.format(pressure))
    st.info('Humidity is:{}'.format(humidity))
    st.info('Zeinth angle is:{}'.format(zeinth))
    st.info('Azimuth angle is :{}'.format(azimuth))
    st.success('The predicted genreated power is {} kw'.format(result))