import streamlit as st
import PyPDF2
import io
import openai
import docx2txt
import pyperclip
import os  
from PyPDF2 import PdfMerger

st.set_page_config(page_title="PDF Question Answerer", page_icon="📄")
st.markdown("""
    <style>
        div[data-baseweb="input"] > div {
            background: rgba(0,0,0,0.1) !important;
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
openai.api_key = st.sidebar.text_input('OpenAI API Key', type='password')
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

def merge_pdf_files(directory, output_filename):
    merger = PdfMerger()
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            merger.append(os.path.join(directory, filename))
    merger.write(output_filename)
    merger.close()

def get_questions_from_gpt(text):
    prompt = text[:4096]
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0.5, max_tokens=30)
    return response.choices[0].text.strip()

def get_answers_from_gpt(text, question):
    prompt = text[:4096] + "\nQuestion: " + question + "\nAnswer:"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0.6, max_tokens=2000)
    return response.choices[0].text.strip()

def main():
    st.title("PDF Question Answerer 📄")
    
    with st.expander("How to use this app 👇"):
     st.write("""
         1. Enter the folder path containing your PDF files.
         2. The app will merge all PDFs into one file.
         3. Ask a question related to the content.
         4. Get an automatically generated answer!
     """)
     
    pdf_folder = st.text_input("Enter the folder path containing PDF files:")
    
    if pdf_folder and os.path.isdir(pdf_folder):
        with st.spinner('Loading...'):
            pdf_files = list_pdf_files(pdf_folder)
        
        if not pdf_files:
            st.warning("No PDF files found in the specified folder.")
        else:
            st.success(f"Number of PDF files found: {len(pdf_files)}")
            
            merged_pdf_filename = "merged.pdf"
            merge_pdf_files(pdf_folder, merged_pdf_filename)
            
            text = extract_text_from_pdf(merged_pdf_filename)
            
            question = get_questions_from_gpt(text)
            st.write("Question: " + question)
            
            user_question1 = "Answer it according to given pdf and in simple words, give answer in minmum 100 words or longer, use extremely simple english, also give dates and other info given in pdf,"
            user_question = st.text_input("Ask a question about the document")
            
            if user_question:
                answer = get_answers_from_gpt(text, user_question1+user_question)
                st.write("Answer: " + answer)
                if st.button("Copy Answer Text"):
                    pyperclip.copy(answer)
                    st.success("Answer text copied to clipboard!")

if __name__ == '__main__':
    main()
