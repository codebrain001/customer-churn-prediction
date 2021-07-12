import streamlit as st
import numpy as np
import pandas as pd

st.title('Customer Churn Prediction App')

st.write("Here's our first attempt at using data to create a table:")

st.subheader("Dataset upload")
data_file = st.file_uploader("Upload CSV", type=["csv"])
if st.button("Process"):
    if data_file is not None:
        #To see file details
        st.write(type(data_file))
        file_detail = {"filename": data_file.name,
        "filetype": data_file.type,
        "filesize": data_file.size}
        st.write(file_detail)
        df = pd.read_csv(data_file)
        st.dataframe(df.head())
