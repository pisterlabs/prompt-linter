import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import docx
import os

load_dotenv()

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
chat_llm = ChatOpenAI(temperature=0.0, request_timeout=120)

def main():
    st.title("Job Description Evaluation")
    
    # Text input for the test
    jd = st.text_input("Enter the Job Description:", value="")

    # Display the test result
    if st.button("Submit"):
        if jd:
            enhanced = enhanced_jd(jd)
            st.write("User Input JD:", jd)
            st.write("Enhanced JD:", enhanced)
            
            if st.button("Old"):
                save_as_docx(jd, "jd.docx")
                
            if st.button("New"):
                save_as_docx(enhanced, "enhanced_jd.docx")

def enhanced_jd(jd):
    title_template = """you are a HR Recruiter bot. you are given a Job description. 
                This is "{topic}" and Score this Job description out of 10. 
                Make some necessary enhancements in the given Job description and only Provide the enhanced Version of the "{topic}".
            """ 
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(topic=jd)
    response = chat_llm(messages)
    save_as_docx(response.content, "enhanced_jd.docx")
    return response.content

def save_as_docx(text, filename):
    doc = docx.Document()
    doc.add_paragraph(text)
    doc.save(filename)

if __name__ == "__main__":
    main()

