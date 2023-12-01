from dotenv import load_dotenv
import os
import streamlit as st
from cohereBot import cohereCall, demoCall
import cohere

co = cohere.Client(st.secrets["COHERE_API_KEY"]) 

#adding the names of upload to a list
def get_names(pdf):
    check_pdf = []
    for uploaded_file in pdf:
        check_pdf.append(uploaded_file.name)
    return check_pdf


# Function to save the PDF to directory
def save_uploadedfile(uploadedfile, path):
     with open(os.path.join("data",path,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to directory".format(uploadedfile.name))


def CoCall(query):

    print("IN COMPONENT")
    with st.spinner('Generating response...'):
        response = co.chat(
            message=query,
            # documents=[],
            model='command',
            temperature=0.2,
            # return_prompt=True,
            chat_history=[],
            connectors=[{"id": "web-search"}],
            prompt_truncation='auto',
        )
        # st.write(response.text)
    return response.text

def function_emma():
    # question = "Who is ELon ? " 
    # print(f"Question: {question}")  # Add this line
    # st.write(question)
    # st.write(cohereCall(question))
    take_prompt = st.text_input("Enter the prompt")

    if st.button("Generate"):
        if len(take_prompt) is 0:
            st.write("Prompt not found")
        else :
            print(take_prompt,"question")
            st.write(cohereCall(take_prompt))


    
def function_condensor ():
    with st.sidebar:
        pdf_s = st.file_uploader("Upload the sylabbus",accept_multiple_files=True,type=['pdf', 'docx','txt'])
        pdf_q = st.file_uploader("Upload the Question paper: Rename question paper to 'year_semester_subject' eg: 2019_1_physics",accept_multiple_files=True,type=['pdf', 'docx'])


        if pdf_s is not None or pdf_q is not None:
            st.write("Syllabus uploaded")
            pdf_s_names = get_names(pdf_s)
            pdf_q_names = get_names(pdf_q)
            st.write(pdf_q_names,pdf_s_names)

        if st.button("Analyze"):
            if pdf_s and pdf_q is not None: 
                for eachFile in pdf_s:
                    save_uploadedfile(eachFile,"syllabus")
                for eachFile in pdf_q:
                    save_uploadedfile(eachFile,"questionPaper")
            elif pdf_s is not None:
                for eachFile in pdf_s:
                    save_uploadedfile(eachFile,"syllabus")
            elif pdf_q is not None:
                for eachFile in pdf_q:
                    save_uploadedfile(eachFile,"questionPaper")
            else :
                st.write("PDF's not Found")
            



def main():

    st.set_page_config(page_title='Upload and get you answer',)
    st.header("Upload and get")

    #Getting the upload

    #two butotns

    if st.button("Syllabus condensor"):
        with st.sidebar:
            pdf_s = st.file_uploader("Upload the sylabbus",accept_multiple_files=True,type=['pdf', 'docx','txt'])
            pdf_q = st.file_uploader("Upload the Question paper: Rename question paper to 'year_semester_subject' eg: 2019_1_physics",accept_multiple_files=True,type=['pdf', 'docx'])


            if pdf_s is not None or pdf_q is not None:
                st.write("Syllabus uploaded")
                pdf_s_names = get_names(pdf_s)
                pdf_q_names = get_names(pdf_q)
                st.write(pdf_q_names,pdf_s_names)
                print(pdf_s_names,pdf_q_names)

            if st.button("Analyze"):
                if pdf_s and pdf_q is not None: 
                    syllabus_content=[]
                    for eachFile in pdf_s:
                        syllabus_content.extend(process_pdf_file(eachFile))
                    for eachFile in pdf_q:
                        save_uploadedfile(eachFile,"questionPaper")
                elif pdf_s is not None:
                    for eachFile in pdf_s:
                        save_uploadedfile(eachFile,"syllabus")
                elif pdf_q is not None:
                    for eachFile in pdf_q:
                        save_uploadedfile(eachFile,"questionPaper")
                else :
                    st.write("PDF's not Found")
        
    if st.button("Chat with EMMA"):
        function_emma()


if __name__ == '__main__':
    main()

    

