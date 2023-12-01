import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import docx
import os
import pdfplumber

load_dotenv()

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
chat_llm = ChatOpenAI(temperature=0.0, request_timeout=120)


def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return str(e)
    


def genrate_qa(en_jd , summary):
    title_template = """you are a Technical interviewer. Develop 15 screening questions for each candidate,
                considering different levels of importance or significance assigned to the "{en_jd}" and the "{summary}"
            """           

    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(en_jd=en_jd,summary=summary)
    response = chat_llm(messages)
    return response.content


    
def main():
    st.title("Screening Round")

    Name = st.text_input("Enter the Name :", value="")
     
    if st.button("Generate Questions"):
        file_path = "enhanced_jd.docx"
        en_jd = read_docx(file_path)
        file_path = "summary.docx"
        summary=read_docx(file_path)
        questions=genrate_qa(en_jd,summary)
        st.text(questions)
    if st.button("Submit"):
        st.markdown("Thank YOu")

if __name__ == "__main__":
    main()

