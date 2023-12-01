import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import tempfile
from llm_helper_function import split_text_q_gen,split_text_q_answer, split_text_docs_vector, extract_text_from_pdf_for_q_gen,extract_text_from_pdf_for_q_answer, create_questions, create_vectordatabase

st.title('🦜🔗 Smart Study Hub')

st.markdown("Smart Study Hub is a tool that helps you to study more efficiently. It generates questions from your study material and answers them for you. This way, you can test your knowledge and learn more effectively.")

# Load env files
# load_dotenv()
# openai_api_key = os.environ.get('OPENAI_API_KEY')

prompt_template = """Use the context below to write an answer to the question.:
    Context: {context}
    Question: {topic}
    Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "topic"]
)

# Initialization of session states
# Since Streamlit always reruns the script when a widget changes, we need to initialize the session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", help="How to get an OpenAI API Key: https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/")
    return input_text

openai_api_key = get_api_key()

with st.container():
    st.markdown("Make sure you've entered your OpenAI API Key. Don't have an API key yet? Read [this](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) article on how to get an API key.")

# Let user upload a file
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

# If user uploaded a file, check if it is a pdf
if uploaded_file is not None:

    if not openai_api_key:
        st.error("Please enter your OpenAI API Key")

    else:
        # Create a LLM
        llm = ChatOpenAI(openai_api_key=openai_api_key,temperature=0.3, model_name="gpt-3.5-turbo-16k")

        if uploaded_file.type == 'application/pdf':

            # Extract and split text from pdf for question generation
            docs_for_q_gen = extract_text_from_pdf_for_q_gen(uploaded_file)

            # Extract and split text from pdf for question answering
            docs_for_q_answer = extract_text_from_pdf_for_q_answer(uploaded_file)

            # Create questions
            if st.session_state['questions'] == 'empty':
                with st.spinner("Generating questions..."):
                    st.session_state['questions'] = create_questions(docs_for_q_gen, llm)

            # Show questions
            st.info(st.session_state['questions'])

            # Create variable for further use of questions.
            questions_var = st.session_state['questions']

            # Split the questions into a list
            st.session_state['questions_list'] = questions_var.split('\n')  # Split the string into a list of questions

            # Create vector database
            # Create the LLM model for the question answering
            llm_question_answer = ChatOpenAI(openai_api_key=openai_api_key,temperature=0.4, model="gpt-3.5-turbo-16k")

            # Create the vector database and RetrievalQA Chain
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.from_documents(docs_for_q_answer, embeddings)
            qa = RetrievalQA.from_chain_type(llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever())


            with st.form('my_form'):
                # Let the user select questions, which will be used to generate answers
                st.session_state['questions_to_answers'] = st.multiselect("Select questions to answer", st.session_state['questions_list'])
                submitted = st.form_submit_button('Generate answers')
                if submitted:
                    # Initialize session state of the answers
                    st.session_state['answers'] = []
                    if 'question_answer_dict' not in st.session_state:
                        # Initialize session state of a dictionary with questions and answers
                        st.session_state['question_answer_dict'] = {}
                    for question in st.session_state['questions_to_answers']:
                        # For each question, generate an answer
                        with st.spinner("Generating answer..."):

                            # Run the chain
                            answer = qa.run(question)
                            st.session_state['question_answer_dict'][question] = answer
                            st.write("Question: ", question)
                            st.info(f"Answer: {answer} ")
                
                    
else:
    st.write("Please upload a pdf file")
    st.stop()