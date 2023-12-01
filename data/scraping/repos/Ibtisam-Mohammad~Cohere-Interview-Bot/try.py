import streamlit as st
import cohere
from PyPDF2 import PdfReader
# uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
# input_text = st.text_input(label='Give the role you are applying for',key="input")
# print('-----------------',input_text,'--------------')
# print(type(input_text))
# if input_text=='': 
#     print('+++++++++++',1)
# if uploaded_file is not None:
#   reader = PdfReader("resume_juanjosecarin.pdf")
#   number_of_pages = len(reader.pages)
#   page = reader.pages
# print(page[0].extract_text())
API_KEY='qKNifTO0EkVWeXAaVzfnnDROVaDZmSbQL5ILgMmc'
co = cohere.Client(API_KEY)
response_resume = co.generate(
      model='command-xlarge-nightly',
      prompt='Hello:',
      max_tokens=10,
      temperature=0,
      k=0,
      p=0.75,
      frequency_penalty=0,
      presence_penalty=0,
      stop_sequences=[],
      return_likelihoods='NONE')
print('response_resume:',response_resume.generations[0].text)