
import streamlit as st
from htmlTemplates import css, bot_template, user_template
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# def ask_question(url, question):
#     try:
#         response = requests.post(url, json={"question": question})
#         if response.status_code == 200:
#             return response.json()["answer"]
#         else:
#             return "Error: Failed to get a response from the chatbot backend."
#     except requests.exceptions.RequestException as e:
#         return f"Error: {str(e)}"
    
def get_website_text(url):
    loader = WebBaseLoader(url)
    index = VectorstoreIndexCreator().from_loaders([loader])
    webDocument = loader.load()
    return webDocument

def get_text_chunks(webDocument):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(webDocument)
    return chunks

def get_vectorstore(document_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_documents(documents=document_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

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
    # Set page config
    st.set_page_config(page_title="Chat with Website",
                        page_icon=":globe_with_meridians:")

    # Diplay CSS styles
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Page header
    # st.title("URL-based Q&A Chatbot")
    
    st.header("Chat with Website :globe_with_meridians:")

    # Input section
    user_question = st.text_input("Ask a question about website:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("User URL Input")
        url = st.sidebar.text_input("Enter the URL:")

        # Process input
        if st.button("Process"):
            if url:
                with st.spinner("Processing"):
                    # Validate the URL format
                    if url.startswith("http://") or url.startswith("https://"):
                    # Load the documents from the URL using WebBaseLoader
                        try:
                            # get pdf text
                            raw_text = get_website_text(url)

                            # get the text chunks
                            text_chunks = get_text_chunks(raw_text)

                            # create vector store
                            vectorstore = get_vectorstore(text_chunks)

                            # Update the processing status in the sidebar
                            st.sidebar.info("Processing completed.")

                            # create conversation chain
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.error("Error: Invalid URL format.")
            else:
                st.error("Please enter URL.")
            
if __name__ == "__main__":
    main()
