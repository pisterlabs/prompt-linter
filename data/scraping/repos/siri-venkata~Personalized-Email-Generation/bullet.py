import os
import openai
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('predictions.csv')
# Load your API key from an environment variable or secret management service
openai.api_key = os.environ['OPENAI_API_KEY']
st.title('Customer Retention Email Generation')
'\n'
customerid = st.selectbox('Select a customer',df['customerid'].unique())
df['gender']=df['gender'].map({'Male':0,'Female':1})
df[['engagementscore','proba']]=(MinMaxScaler().fit_transform(df[['engagementscore','proba']])*100).astype(int)
df = df[df['customerid']==customerid]

st.header('Customer Profile')
st.write(df.T)
st.header('Configuration options')
emails = ['''1. subject is:  new product promotion experience
2. Make the valued customer aware of special promotions with exclusive discounts.
3. Big savings promotions are only available to our customers for a limited time.
4. Make use of our discounts and contact us if you have questions
''',
'''1. Subject: Enjoy [Discount]% Off Our New Products 
2. Since the customer is valued they got an offer of [Discount]%.
3. Big savings promotions are only available to our customers for a limited time.
4. Make use of our discounts and contact us if you have questions
''',
'''1. Subject: Get Ready to Experience Our Exclusive Discount Offer
2. [Link Here] provides an exclusive offer to the valued customer. 
3. Big savings promotions are only available to our customers for a limited time.
4. Make use of our discounts and contact us if you have questions
''']
st.write('Email 1')
emails[0] = st.text_area('Write the bullet points to include in the email.',value=emails[0])
email1_threshold = st.slider('Select a range for churn probability',0,100,(25,50))
st.write('Email 2')
emails[1] = st.text_area('Write the bullet points to include in the email.',value=emails[1])
email2_threshold = st.slider('Select a range for churn probability',0,100,(50,75))
st.write('Email 3')
emails[2] = st.text_area('Write the bullet points to include in the email.',value=emails[2])
email3_threshold = st.slider('Select a range for churn probability',0,100,(75,100))

proba = int(df['proba'].values[0])
'Discount'
discount = st.slider('What discount are we willing to offer?', 0, 100, 5)


if st.button('Generate Email'):
	text = emails[0] if proba in range(*email1_threshold) else emails[1] if proba in range(*email2_threshold) else emails[2] if  proba in range(*email3_threshold) else ''
	if text!='':
		prompt='''Instruction: Write an email in minimum 400 words with information provided in the points below. Strictly adhere to minimum word limits. Address the email to {}.discount percentage to offer is {}%.\n'''.format(customerid,discount,)
		prompt+="Content: "+text
		prompt+='\n400 word Email:\n'
		response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.4, max_tokens=3500)
		mail = response['choices'][0]['text']
		st.header('Email')
		st.write(mail)
		print(len(mail.split()))
	else:
		st.write('No Action Needed')

