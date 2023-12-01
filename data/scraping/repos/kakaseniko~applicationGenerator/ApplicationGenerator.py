import streamlit as st
from langchain.llms import Cohere
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text


llm = Cohere(cohere_api_key="Q37DUZNj7vmTC4HYjFrQ1yNUDAH5PneuNJ5iSwpK")

def truncate_at_sentence(text):
    sentences = text.split('.')
    filtered_sentences = [sentence for sentence in sentences if not sentence.strip().endswith('?')]
    truncated_text = '.'.join(filtered_sentences)
    return truncated_text

st.write("""
# Job application generator 📄
**Enter the job description and upload your CV to generate an application letter.**
         \n""")

with st.form('my_form'):
  text = st.text_area('Job description:', '')
  uploaded_file = st.file_uploader("CV as a PDF file:", type="pdf")
  submitted = st.form_submit_button('Submit')
  
  if submitted:
    with st.spinner('Generating application letter...'):
          if uploaded_file is not None:
            pdf_text = extract_text(uploaded_file)
            doc = Document(page_content=text)
            answer = llm(prompt="write a job application for the following vacancy: " + doc.page_content + "based on this cv:" + pdf_text, max_tokens = 1000)
            sentences = answer.split(':')
            updated_answer = ':'.join(sentences[1:])
            updated_answer = truncate_at_sentence(updated_answer)

            st.write(updated_answer)
            st.success("Application letter generated successfully! To regenerate, click 'Submit' again.")

            