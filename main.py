
# import streamlit as st
# import pandas as pd
# import numpy as np

# def predict(model, input_df):
# 	predictions_df = predict_model(estimator=model, data=input_df)
# 	predictions = predictions_df['Label'][0]
# 	return predictions


# def main():
# 	add_selectbox = st.sidebar.selectbox(
# 	"How would you like to predict?",
# 	("Online", "Batch"))
# 	st.sidebar.info('This app is created to predict Customer Churn')
# 	st.sidebar.image(image2)
# 	st.title("Predicting Customer Churn")
# 	if add_selectbox == 'Online':
# 		state =st.selectbox('letter code of the US state of customer residence :',['','AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA','ID',\
# 		'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',\
# 		'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV','WY'])
# 		account_length=st.number_input('Number of months the customer has been with the current telco provider :' , min_value=0, max_value=240, value=0)
# 		area_code=st.selectbox('"area_code_AAA" where AAA = 3 digit area code :' , ['','area_code_408', 'area_code_415', 'area_code_510'])
# 		international_plan=st.selectbox('The customer has international plan :' , ['','yes', 'no'])
# 		voice_mail_plan=st.selectbox('The customer has voice mail plan :' , ['','yes', 'no'])
# 		number_vmail_messages=st.slider('Number of voice-mail messages. :' , min_value=0, max_value=60, value=0)
# 		total_day_minutes=st.slider('Total minutes of day calls :' , min_value=0, max_value=360, value=100)
# 		total_day_calls=st.slider('Total day calls :' , min_value=0, max_value=200, value=50)
# 		total_eve_minutes=st.slider('Total minutes of evening calls :' , min_value=0, max_value=400, value=200)
# 		total_eve_calls=st.slider('Total number of evenig calls :' , min_value=0, max_value=200, value=100)
# 		total_night_minutes=st.slider('Total minutes of night calls :' , min_value=0, max_value=400, value=200)
# 		total_night_calls=st.slider('Total number of night calls :' , min_value=0, max_value=200, value=100)
# 		total_intl_minutes=st.slider('Total minutes of international calls :' , min_value=0, max_value=60, value=0)
# 		total_intl_calls=st.slider('Total number of international calls :' , min_value=0, max_value=20, value=0)
# 		number_customer_service_calls=st.slider('Number of calls to customer service :' , min_value=0, max_value=10, value=0)
# 		output=""
# 		input_dict={'state':state,'account_length':account_length,'area_code':area_code,'international_plan':international_plan,'voice_mail_plan':voice_mail_plan\
# 		,'number_vmail_messages':number_vmail_messages,'total_day_minutes':total_day_minutes,'total_day_calls':total_day_calls\
# 		,'total_eve_minutes':total_eve_minutes,'total_eve_calls':total_eve_calls,'total_night_minutes':total_night_minutes\
# 		,'total_night_calls':total_night_calls,'total_intl_minutes':total_intl_minutes,'total_intl_calls':total_intl_calls\
# 		,'number_customer_service_calls':number_customer_service_calls}
# 		input_df = pd.DataFrame([input_dict])
# 		if st.button("Predict"):
# 			output = predict(model=model, input_df=input_df)
# 			output = str(output)
# 		st.success('Churn : {}'.format(output))
# 	if add_selectbox == 'Batch':
# 		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
# 		if file_upload is not None:
# 			data = pd.read_csv(file_upload)
# 			predictions = predict_model(estimator=model,data=data)
# 			st.write(predictions)
# if __name__ == '__main__':
# 	main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from PIL import Image


# st.title('Telco Customer Churn Prediction App')


# def preprocessing():
#     pass

# def predict():
#     pass


# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == 'SVM':
#         C = st.sidebar.slider('C', 0.01, 10.0)
#         params['C'] = C
#     elif clf_name == 'KNN':
#         K = st.sidebar.slider('K', 1, 15)
#         params['K'] = K
#     else:
#         max_depth = st.sidebar.slider('max_depth', 2, 15)
#         params['max_depth'] = max_depth
#         n_estimators = st.sidebar.slider('n_estimators', 1, 100)
#         params['n_estimators'] = n_estimators
#     return params

# params = add_parameter_ui(classifier_name)

# def get_classifier(clf_name, params):
#     clf = None
#     if clf_name == 'SVM':
#         clf = SVC(C=params['C'])
#     elif clf_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     else:
#         clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
#             max_depth=params['max_depth'], random_state=1234)
#     return clf

# clf = get_classifier(classifier_name, params)

# #### CLASSIFICATION ####

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test, y_pred)

# st.write(f'Classifier = {classifier_name}')
# st.write(f'Accuracy =', ac


# def main():
#     image = Image.open('App.jpg')
#     add_selectbox = st.sidebar.selectbox(
# 	"How would you like to predict?", ("Batch", "Online"))
#     st.sidebar.info('This app is created to predict Customer Churn')
#     st.sidebar.image(image)
   
#     if add_selectbox == 'Batch':
#         file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
#         if file_upload is not None:
#             data = pd.read_csv(file_upload)
#     else:
#         pass
#     pass


# if __name__ == '__main__':
# 	main()




  
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
# import seaborn as sns
# import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



st.write("""
# Churn Prediction App
Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers
who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with
the company. 
This app predicts the probability of a customer churning using Telco Customer data. Here
customer churn means the customer does not make another purchase after a period of time. 
""")



df_selected = pd.read_csv("data/churn.csv")
df_selected_all = df_selected[['gender', 'Partner', 'Dependents', 'PhoneService', 
                                     'tenure', 'MonthlyCharges', 'Churn']].copy()
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)


uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('gender',('Male','Female'))
        PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
        tenure = st.sidebar.slider('tenure', 0.0,72.0, 0.0)

        data = {'gender':[gender], 
                'PaymentMethod':[PaymentMethod], 
                'MonthlyCharges':[MonthlyCharges], 
                'tenure':[tenure],}
        
        features = pd.DataFrame(data)
        return features
    input_df = user_input_features()



churn_raw = pd.read_csv('data/churn.csv')




churn_raw.fillna(0, inplace=True)
churn = churn_raw.drop(columns=['Churn'])
df = pd.concat([input_df,churn],axis=0)




encode = ['gender','PaymentMethod']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)


features = ['MonthlyCharges', 'tenure', 'gender_Female', 'gender_Male',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

df = df[features]



# Displays the user input features
st.subheader('User Input features')
print(df.columns)
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
churn_labels = np.array(['No','Yes'])
st.write(churn_labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

