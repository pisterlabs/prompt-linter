import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import time 
import os
# Define the CSS style for the "Get Started" button
button_style = """
    background-color: red;
    color: white;
    font-size: 16px;
    padding: 10px;
    border: none;
    border-radius: 5px;
"""


contact_form = f"""
    <form action="">
        <button type="submit">Get Started</button>
    </form>
"""

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def get_pdf_text(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat avec multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "started" not in st.session_state:
        st.session_state.started = False

    # Landing page with "Get Started" button
    if not st.session_state.started:
    # Display the banner at the top of the page
        st.markdown("# Welcome to the Landing Page")
        st.markdown("Click the button below to get started.")
        st.image("./doc/Marsa Maroc Concour de Recrutement 5 Postes.png", use_column_width=True)  # Replace "path_to_banner_image" with the path to your banner image

        pdf_docs = "C:\\Users\\Hinnovis\\Desktop\\homework_data\\projet_pfa\\chat_bot\\pdf"

        # Use the html component to apply the custom CSS style to the button
        local_css("style.css")
        st.markdown(contact_form,  unsafe_allow_html=True)
        time.sleep(3)
        if st.form("Get Started"):
            st.write("##")
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

            # Set the session state to indicate that the "Get Started" button is clicked
            st.session_state.started = True
            # Rerun the Streamlit app to show the main page content
            st.experimental_rerun()


    # Main page content
    else:
        st.header("Chat avec multiple PDFs :books:")

        # Container for user input in the sidebar
        st.sidebar.subheader("User Input")
        
        # Add a button to clear the user input
        user_question = st.sidebar.text_area("Poser la question à propos de vos documents:", key="textarea_key")

        col1, col2 = st.sidebar.columns(2)

        # Add the buttons in the respective columns
        with col1:
            submit_button = st.button('Entrer')

        if submit_button:
            handle_userinput(user_question)

if __name__ == "__main__":
    main()