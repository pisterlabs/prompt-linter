import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from reportlab.pdfgen import canvas
from io import BytesIO
import base64
import re
import time





# method to convert uploaded file in pdf format to a string text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



# returns chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# returns chunks without overlap
# def get_text_chunks_wo_overlap(text):
#     text_splitter = CharacterTextSplitter(
#         separator = "\n",
#         chunk_size = 400,
#         chunk_overlap = 0,
#         length_function = len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# method to generate keywords from openai llm model
# def generate_keywords(text, api_key):
#     try:
#         openai.api_key = api_key  # Set the API key
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=text,
#             max_tokens=int(len(text)/10)  # Adjust max_tokens as needed for your desired length
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         st.error(f"Error occurred: {e}")
#         return None


# method to create vector store and store chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# method that returns the chain of user inputs and generated outputs from vector store
def get_conversation_chain(vectorstore):
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


# method to output the user question, the generated response, the summary, and the questions
@st.cache_data
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'][1:]

    # chat display
    for i, message in enumerate(st.session_state.chat_history[3:]):
        cleaned_message = str(message)[:-1].replace('content=\'', '').strip()
        if i % 2 == 0:
            st.markdown(
                f'<div style="background-color: #e6f2ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{cleaned_message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>{cleaned_message}</strong> </div>',
                unsafe_allow_html=True
            )


# method to generate summary from the uploaded file
@st.cache_data
def generate_summary(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    summary_messages = []
    for i, message in enumerate(st.session_state.chat_history):
        cleaned_message = str(message)[:-1].replace('content=\'', '').strip()
        if i % 2 != 0:
            summary_messages.append(cleaned_message)
    return summary_messages


# carry on from get_summary method
def get_document_summary():
    if st.session_state.conversation:
        user_question = "Provide a summary of the uploaded file. If the file is a textbook or a book, disregard the table of contents and generate a brief overview of the primary contents of the book or chapters in the book."
        return generate_summary(user_question)
    else:
        st.warning("Please upload a document first.")



# method to generate questions from the uploaded file
@st.cache_data
def get_questions(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    question_output = []
    for i, question in enumerate(st.session_state.chat_history):
        cleaned_message = str(question)[:-1].replace('content=\'', '').strip()
        if i % 2 != 0:
            if cleaned_message.endswith('?'):
                cleaned_message = cleaned_message.replace('\n', ' ')
                question_output.append(cleaned_message)
            else:
                cleaned_message = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_message)  # Remove special characters
                cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()  # Remove extra spaces
                cleaned_message = cleaned_message.capitalize()  # Capitalize the sentence
                cleaned_message = cleaned_message.replace('\n', ' ')
                question_output.append(f"{cleaned_message}?")
    return question_output









# carry on from get_questions method
def get_document_questions():
    if st.session_state.conversation:
        user_question = "Generate a list of detailed questions about the content contained within uploaded document. If the file is a textbook or a book, disregard the table of contents and generate questions based on the primary contents of the book or chapters in the book. WRITE THE QUESTIONS IN QUESTION FORMAT (e.g What does this topic mean?) for the user to practice about the uploaded document. DO NOT INCLUDE THE CHARACTERS \n WITHIN THE QUESTIONS! Always end each question with a question mark!"
        return get_questions(user_question)
    else:
        st.warning("Please upload a document first.")




# main method
def main():
    load_dotenv()
    st.set_page_config(
        page_title = "NoteFinder",
        page_icon = ":notebook:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    col1, col2 = st.columns([1,2])
    with col1:
        st.title("NoteFinder")
    with col2:
        st.image("/Users/sidsomani/Desktop/streamlit_projects/pages/logovs2.png", width = 80)

    for i in range(4):
        st.text("")


    with st.sidebar:
        st.image("/Users/sidsomani/Desktop/streamlit_projects/pages/logovs2.png", width = 290)

        for i in range(2):
            st.text("")

        openai_api_key = st.session_state.openai_api_key
        for i in range(2):
            st.text("")

        pdf_docs = st.file_uploader("Upload your files", accept_multiple_files=True)

        for i in range(2):
            st.text("")

        if pdf_docs and st.button("Process"):
            st.text("")
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # for chunk in text_chunks:
                #     keywords = generate_keywords(chunk, openai_api_key)
                #     # st.write(keywords)
                #     keywords_combined+=keywords


                # create vector store with embedded chunks
                vectorstore = get_vectorstore(text_chunks)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)



    tab1, tab2, tab3 = st.tabs(["Finder", "Summary", "Questions"])
    with tab1:
        st.text("")
        user_question = st.text_input("Ask a question about your documents")
        if user_question:
            handle_userinput(user_question)

    with tab2:
        st.text("")
        summary = get_document_summary()
        if summary:
            st.session_state.summary_message = summary
            for message in st.session_state.summary_message:
                st.markdown(
                    f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{message}</div>',
                    unsafe_allow_html=True
                )

    with tab3:
        st.text("")
        questions = get_document_questions()
        if questions:
            st.session_state.question_bullet = questions
            for question in st.session_state.question_bullet[1:]:
                st.markdown(
                    f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{question}</div>',
                    unsafe_allow_html=True
                )





if __name__ == '__main__':
    main()



