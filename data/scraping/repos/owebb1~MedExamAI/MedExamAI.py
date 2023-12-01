
import os
import dotenv
import streamlit as st

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate


def main():
    dotenv.load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    st.title("MedExamAI")

    with st.container():
        st.markdown(
            """
            ## Medical Exam Question Generator
            In this Streamlit application, we are demonstrating the power of AI in creating a personalized and adaptive study aid for medical school students.

            MedExamAI is a unique tool designed for medical students who are looking for high-quality practice exam questions and personalized study plans. It uses artificial intelligence to analyze your performance and tailor study plans to your specific needs.

            Here's how MedExamAI works:

            The application generates 30 exam questions, providing users with a robust sample of potential questions they might encounter in their studies or exams.
            These questions are designed with medical school students in mind, and they mirror the complexity and structure of actual test questions.
            After users complete the questions, MedExamAI uses artificial intelligence to analyze the responses and determine which questions were answered incorrectly.
            The application then uses these insights to create a customized study plan. This plan focuses on the areas where the user had the most difficulty, ensuring that their study time is used as effectively as possible.

            Keep an eye on my Twitter for explanations of the code and other updates about MedExamAI.
            """
        )
    with st.container():
        if st.button("Generate Your Practice Step 1 Exam"):
            with st.spinner("Generating Exam..."):
                template = """
                As an expert in medical education, generate 30 practice questions from the USCME Step 1 Exam.

                The practice questions should cover a wide range of topics and difficulty levels, reflecting the content and format of the actual exam.

                Please provide clear and concise stem statements, followed by multiple-choice options. Each question should have a single correct answer that is listed in a list of answers at the end of the questioning.

                Return the questions and answers in markdown format.

                Examples:

                ### Question 1:

                A 45-year-old male presents with chest pain and shortness of breath. On physical examination, he has decreased breath sounds on the left side. A chest X-ray reveals a complete collapse of the left lung. What is the most likely diagnosis?

                A) Pneumonia

                B) Pulmonary embolism

                C) Pleural effusion

                D) Tension pneumothorax

                ### Question 2:

                A 32-year-old female presents with fatigue, weight gain, and cold intolerance. On physical examination, she has a slow heart rate and dry skin. Laboratory tests reveal elevated thyroid-stimulating hormone (TSH) levels and low free thyroxine (T4) levels. What is the most likely diagnosis?

                A) Graves' disease

                B) Hashimoto's thyroiditis

                C) Thyroid storm

                D) Subacute thyroiditis

                ...

                Question 30:

                A 60-year-old male presents with progressive memory loss, confusion, and difficulty performing daily activities. On physical examination, he has impaired judgment and personality changes. Imaging studies reveal cortical atrophy and enlarged ventricles. What is the most likely diagnosis?

                A) Alzheimer's disease

                B) Vascular dementia

                C) Lewy body dementia

                D) Frontotemporal dementia

                ### Answers:
                1: D) Tension pneumothorax

                2: B) Hashimoto's thyroiditis

                ...

                30: A) Alzheimer's disease
                """

                prompt_template = PromptTemplate(
                    input_variables=[], template=template)

                # Initialize the language model. This is the model that will generate the output.
                llm = ChatOpenAI(
                    client="gpt-4",
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0.2
                )

                # Initialize the LLMChain.
                llm_chain = LLMChain(
                    llm=llm,
                    prompt=prompt_template
                )

                practice_test = llm_chain.predict()
                st.markdown(practice_test)


if __name__ == "__main__":
    main()
