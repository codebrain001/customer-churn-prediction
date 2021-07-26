import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image


def preprocessing():
    pass


def predict():
    pass



def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')

    #Setting Application header
    st.markdown("""
    This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both batch data prediction and online prediction. \n
    :dart: The batch predictions will utilize three machine learning algorithms: Random Forest Classifer, Support Vector Classifier and XGB classifer. \n
    :dart: The online prediction will accept inputs and use filters in the sidebar. \n
    :dart: The online prediction is backed with a Random Forest model \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Batch", "Online"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)
   

    if add_selectbox == "Batch":
        st.write("Batch prediction sequence")
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
    else:
        st.write("Online prediction")

if __name__ == '__main__':
	main()



