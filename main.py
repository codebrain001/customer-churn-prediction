# import streamlit as st
# import numpy as np
# import pandas as pd

# st.title('Customer Churn Prediction App')

# st.write("Here's our first attempt at using data to create a table:")

# st.subheader("Dataset upload")
# data_file = st.file_uploader("Upload CSV", type=["csv"])
# if st.button("Process"):
#     if data_file is not None:
#         #To see file details
#         st.write(type(data_file))
#         file_detail = {"filename": data_file.name,
#         "filetype": data_file.type,
#         "filesize": data_file.size}
#         st.write(file_detail)
#         df = pd.read_csv(data_file)
#         st.dataframe(df.head())


import streamlit as st
import pandas as pd
import numpy as np

def predict(model, input_df):
	predictions_df = predict_model(estimator=model, data=input_df)
	predictions = predictions_df['Label'][0]
	return predictions


def main():
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict Customer Churn')
	st.sidebar.image(image2)
	st.title("Predicting Customer Churn")
	if add_selectbox == 'Online':
		state =st.selectbox('letter code of the US state of customer residence :',['','AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA','ID',\
		'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',\
		'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV','WY'])
		account_length=st.number_input('Number of months the customer has been with the current telco provider :' , min_value=0, max_value=240, value=0)
		area_code=st.selectbox('"area_code_AAA" where AAA = 3 digit area code :' , ['','area_code_408', 'area_code_415', 'area_code_510'])
		international_plan=st.selectbox('The customer has international plan :' , ['','yes', 'no'])
		voice_mail_plan=st.selectbox('The customer has voice mail plan :' , ['','yes', 'no'])
		number_vmail_messages=st.slider('Number of voice-mail messages. :' , min_value=0, max_value=60, value=0)
		total_day_minutes=st.slider('Total minutes of day calls :' , min_value=0, max_value=360, value=100)
		total_day_calls=st.slider('Total day calls :' , min_value=0, max_value=200, value=50)
		total_eve_minutes=st.slider('Total minutes of evening calls :' , min_value=0, max_value=400, value=200)
		total_eve_calls=st.slider('Total number of evening calls :' , min_value=0, max_value=200, value=100)
		total_night_minutes=st.slider('Total minutes of night calls :' , min_value=0, max_value=400, value=200)
		total_night_calls=st.slider('Total number of night calls :' , min_value=0, max_value=200, value=100)
		total_intl_minutes=st.slider('Total minutes of international calls :' , min_value=0, max_value=60, value=0)
		total_intl_calls=st.slider('Total number of international calls :' , min_value=0, max_value=20, value=0)
		number_customer_service_calls=st.slider('Number of calls to customer service :' , min_value=0, max_value=10, value=0)
		output=""
		input_dict={'state':state,'account_length':account_length,'area_code':area_code,'international_plan':international_plan,'voice_mail_plan':voice_mail_plan\
		,'number_vmail_messages':number_vmail_messages,'total_day_minutes':total_day_minutes,'total_day_calls':total_day_calls\
		,'total_eve_minutes':total_eve_minutes,'total_eve_calls':total_eve_calls,'total_night_minutes':total_night_minutes\
		,'total_night_calls':total_night_calls,'total_intl_minutes':total_intl_minutes,'total_intl_calls':total_intl_calls\
		,'number_customer_service_calls':number_customer_service_calls}
		input_df = pd.DataFrame([input_dict])
		if st.button("Predict"):
			output = predict(model=model, input_df=input_df)
			output = str(output)
		st.success('Churn : {}'.format(output))
	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			predictions = predict_model(estimator=model,data=data)
			st.write(predictions)
if __name__ == '__main__':
	main()
