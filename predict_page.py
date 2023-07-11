import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
image = Image.open('FuelPredx_logo.jpg')

def load_model():
    with open('gbr_saved_steps (1).pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]

def show_predict_page():

    st.markdown("<h1 style='text-align: center; color: DarkGoldenRod;'>FuelPredx</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: DarkGreen;'>A cloud based serverless application for fuel consumption prediction of 8hp- 48hp tractors using GBR ML model </h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: DarkRed;'>Input the following tractor parameters for predicting the fuel consumption: </h3>", unsafe_allow_html=True)

    st.sidebar.image(image, use_column_width=True)

    Tractor_PTO = st.number_input('Tractor maximum PTO power (hp)')
    Tractor_PTO2 = Tractor_PTO/1.36
    Engine_Speed = st.number_input('Engine operating speed (rpm)')
    Speed_Depression = st.number_input('Engine speed depression (rpm)')

    ok = st.button("Predict fuel consumption")
    if ok:
        X = np.array([[Tractor_PTO2, Engine_Speed, Speed_Depression]])
        #Y = sc.transform(X)
        #Y = Y.astype(float)
        X = X.astype(float)
        salary = regressor.predict(X)
        st.subheader(f"The estimated Fuel Consumption(L/h) is {salary[0]:.2f}")



