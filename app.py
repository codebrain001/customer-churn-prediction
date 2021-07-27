import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image


#Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Machine Learning operations
from sklearn.model_selection import train_test_split

def preprocessing(df):

    pass


def predict():
    pass



def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')

    #Setting Application header
    st.markdown("""
     :dart: This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both batch data prediction and online prediction. \n
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
        st.subheader("Dataset upload")
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            st.write(type(file_upload))
            file_detail = {"filename": file_upload.name,
            "filetype": file_upload.type,
            "filesize": file_upload.size}
            st.write(file_detail)
            data_df = pd.read_csv(file_upload)
        
        
        classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('Logistic Regression', 'SVM', 'Random Forest')
)


    else:
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0 )
        gender = st.selectbox('Gender:', ['Male', 'Female'])
        partner = st.selectbox('Partner:', ['Yes', 'No'])
        dependents = st.selectbox('Dependent:', ['Yes', 'No'])
        seniorcitizen = st.selectbox('Senior Citizen:', ['Yes', 'No'])
        phoneservice = st.selectbox('Phone Service:', ['Yes', 'No'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
        PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0 )

        data = {'gender':[gender], 
                'SeniorCitizen': [seniorcitizen],
                'partner': [partner],
                'Dependents': [dependents],
                'tenure':[tenure],
                'PhoneSevice': [phoneservice],
                'Contract': [contract],
                'PaperlessBilling': [paperlessbilling],
                'PaymentMethod':[PaymentMethod], 
                'MonthlyCharges':[monthlycharges], 
                }

        features_df = pd.DataFrame(data)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
           

        st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.button('Predict Churn')
if __name__ == '__main__':
	main()


