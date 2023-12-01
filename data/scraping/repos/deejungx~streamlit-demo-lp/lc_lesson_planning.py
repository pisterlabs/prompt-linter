import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


prompt = PromptTemplate.from_template(
    """As an expert in {grade} {subject}, your task is to generate a lesson plan using
    the 5E instructional model for {topic}. The lesson duration is {lesson_duration} minutes.
    The lesson plan should follow the format provided below:
        ```
        Title: \n
        Objective: \n
        Materials: \n
        Procedure: \n
            - Engage (N mins):
            - Explore (N mins):
            - Explain (N mins):
            - Elaborate (N mins):
            - Evaluate (N mins):
        ```
        """
)

st.title("🦜🔗 Lesson Plan Generator")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


def generate_response(grade, subject, topic, lesson_duration):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    prmpt = prompt.format(
        grade=grade, subject=subject, topic=topic, lesson_duration=lesson_duration
    )
    st.info(llm(prmpt))


with st.form("my_form"):
    "Generate 5E Lesson Plan"
    grade = st.number_input("Grade", 1, 10, 8)
    subject = st.text_input("Subject", "Science")
    topic = st.text_input("Topic", "Heredity")
    lesson_duration = st.slider("Lesson duration (in Minutes)", 0, 150, 60)
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        generate_response(grade, subject, topic, lesson_duration)
