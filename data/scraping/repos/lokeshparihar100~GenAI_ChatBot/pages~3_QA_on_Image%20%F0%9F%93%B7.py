import streamlit as st
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import pytesseract
from PIL import Image

def get_pdf_text(image_docs):
    text = "This text is generated from a image."
    
    for image in image_docs:
        # windows
        # pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
        try:
            img = Image.open(image)
            data = pytesseract.image_to_string(img)
            data.strip()
            text=text+data
        except:
            st.write("Not valid image")

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
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

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
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Image", page_icon=":camera:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with Image :camera:")
    user_question = st.text_input("Ask a question about your Image:")
    if user_question:
        handle_userinput(user_question)
        
    with st.sidebar:
            
        st.subheader("Your Images")
        image_docs = st.file_uploader("Upload your Images here and click on 'Process'", accept_multiple_files=True, type=['png', 'jpeg', 'jpg'])
        
        if st.button("Process"):
            if image_docs:
                with st.spinner("Processing"):
                    try:
                        # get pdf text
                        raw_text = get_pdf_text(image_docs)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Update the processing status in the sidebar
                        st.sidebar.info("Processing completed.")

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.error("Plase upload an image.")
                
if __name__ == '__main__':
    main()