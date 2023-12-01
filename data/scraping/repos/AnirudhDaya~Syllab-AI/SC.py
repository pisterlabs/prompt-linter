import streamlit as st
import pandas as pd
import os
import time
from cohereBot import cohereCall
from dotenv import load_dotenv
import extraction as ex
load_dotenv()
os.environ['cohere_api_key'] = os.getenv('cohere_api_key')



# Fucntion to save the PDF to directory


def get_names(pdf):
     #adding the names of upload to a list
    check_pdf = []
    for uploaded_file in pdf:
        check_pdf.append(uploaded_file.name)


def main():

    # load_dotenv()
    st.set_page_config(page_title='Syllabus GPT',)
    st.header("Give your prompt based on the Uploaded Docs")

    #getting the upload 
    with st.sidebar:
        st.header("Upload the PDF's")
        pdf_s = st.file_uploader("Upload the sylabbus",accept_multiple_files=True,type=['pdf', 'docx','txt'])
        pdf_q = st.file_uploader("Upload the Question paper: Rename question paper to 'year_semester_subject' eg: 2019_1_physics",accept_multiple_files=True,type=['pdf', 'docx'])
        

        pdf_s_names = get_names(pdf_s)
        pdf_q_names = get_names(pdf_q)

        
        # print(len(pdf_s_names))
        st.write(pdf_q_names,pdf_s_names)
        #TODO : use pyPDF to extract the file and send to extract with out saving.


        if st.button("Analyze"):
            syllabus_content=[]
            question_content=[]
            if pdf_s and pdf_q is not None: 
                for eachFile in pdf_s:
        
                    syllabus_content.append(ex.process_pdf_contents(eachFile))
                for eachFile in pdf_q:
                 
                    question_content.append(ex.process_pdf_contents(eachFile))
            if pdf_s is not None:
                for eachFile in pdf_s:
                   
                    syllabus_content.append(ex.process_pdf_contents(eachFile))
            elif pdf_q is not None:
                for eachFile in pdf_q:
                   
                    question_content.append(ex.process_pdf_contents(eachFile))
             
            if len(pdf_s) == 0 or len(pdf_q) == 0:
                st.write("PDF's not Found") 
            else:
                print("","condense")
                # print("syllabus")
                # print(syllabus_content)
                # print("question")
                # print(len(question_content))
                cohereCall(syllabus_content,question_content)
                # demoCall()
               
    take_prompt = st.text_input("Enter the prompt")
    st.write(take_prompt) 

    if st.button("Generate"):
        if len(take_prompt) is 0:
            st.write("Prompt not found")
        else :
            print(take_prompt,"question")
            cohereCall(take_prompt,"question")


def delete_files():
    file_list = os.listdir("./data/syllabus")
    for file_name in file_list:
        file_path = os.path.join("./data/syllabus", file_name)
        os.remove(file_path)
    file_list = os.listdir("./data/questionPaper")
    for file_name in file_list:
        file_path = os.path.join("./data/questionPaper", file_name)
        os.remove(file_path)


if __name__ == '__main__':
    main()
    # time.sleep(45)
    # delete_files()
    