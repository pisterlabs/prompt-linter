import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pymongo
import os
from dotenv import load_dotenv
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/K6RJmd9/t-l-chargement.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
print("Existing Environment Variables:", os.environ)
load_dotenv(".env")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Function to store and retrieve past chat sessions
def save_chat_history(chat_history):
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = []
    st.session_state.chat_sessions.append(chat_history)

def get_saved_chat_sessions():





    client = pymongo.MongoClient("mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")
    db = client["chatbotesprit"]
    chat_sessions_collection = db["chat_sessions"]

    # Retrieve chat history for the selected chatbot
    chat_history = chat_sessions_collection.find_one({"bot_name": st.session_state.selected_bot})

    client.close()

    # If chat history exists, return it; otherwise, return an empty list
    return chat_history.get("chat_history", []) if chat_history else []


# Function to display the sidebar with chat history
def display_chat_history_sidebar():
    # Initialize the 'chat_sessions' session state variable if it doesn't exist
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}

    chat_sessions = get_saved_chat_sessions()
    if chat_sessions:
        st.sidebar.subheader("Past Chat Sessions")
        for i, chat_history in enumerate(chat_sessions):
            session_btn = st.sidebar.button(f"Chat Session {i+1}")
            if session_btn:
                st.session_state.conversation = None
                st.session_state.chat_history = chat_history

    # Display the stored pdf_chunks for the selected chatbot
    if st.session_state.selected_bot:
        pdf_chunks = get_stored_pdf_chunks(st.session_state.selected_bot)
        if pdf_chunks:
            st.sidebar.subheader("Stored PDF Chunks")
            for i, chunk in enumerate(pdf_chunks):
                pdf_chunk_btn = st.sidebar.button(f"PDF Chunk {i+1}")
                if pdf_chunk_btn:
                    st.session_state.selected_pdf_chunk = chunk



def get_created_chatbots():
    client = pymongo.MongoClient("mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")
    db = client["chatbotesprit"]
    chatbots_collection = db["chatbots"]
    created_chatbots = chatbots_collection.find({}, {"_id": 0, "bot_name": 1})
    chatbot_names = [chatbot["bot_name"] for chatbot in created_chatbots]
    client.close()  # Close the connection after fetching data
    return chatbot_names

def handle_chatbot_creation(new_bot_name):
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MongoDB connection string not found. Please set the MONGODB_URI environment variable.")
        
    client = pymongo.MongoClient("mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")
    db = client["chatbotesprit"]
    chatbots_collection = db["chatbots"]

    chatbot_entry = {"bot_name": new_bot_name}
    chatbots_collection.insert_one(chatbot_entry)

    client.close()

    st.session_state.selected_bot = new_bot_name

def store_pdf_text_chunks(bot_name, text_chunks):
    client = pymongo.MongoClient("mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")
    db = client["chatbotesprit"]

    # Create or get the PDF chunks collection (replace "pdf_chunks" with your preferred collection name)
    pdf_chunks_collection = db["pdf_chunks"]

    # Create a new entry for the chatbot in the PDF chunks collection
    pdf_chunks_entry = {"bot_name": bot_name, "text_chunks": text_chunks}
    pdf_chunks_collection.insert_one(pdf_chunks_entry)

    client.close()


def get_conversation_chain(vectorstore):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_stored_pdf_chunks(bot_name):
    client = pymongo.MongoClient("mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")
    db = client["chatbotesprit"]
    pdf_chunks_collection = db["pdf_chunks"]

    # Retrieve the stored pdf_chunks for the selected chatbot
    pdf_chunks_entry = pdf_chunks_collection.find_one({"bot_name": bot_name})
    client.close()

    # If pdf_chunks entry exists, return it; otherwise, return an empty list
    return pdf_chunks_entry.get("text_chunks", []) if pdf_chunks_entry else []
def save_pdf_chunks_to_db(bot_name, text_chunks):
    # Connect to the MongoDB server
    client = pymongo.MongoClient( "mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")

    db = client["chatbotesprit"]
    pdf_chunks_collection = db["pdf_chunks"]

    # Save the PDF chunks for the selected bot
    pdf_chunks_collection.update_one(
        {"bot_name": bot_name},
        {"$set": {"text_chunks": text_chunks}},
        upsert=True  # Create a new entry if not found
    )

    client.close()

def save_chat_history_to_db(bot_name, chat_history):
    # Connect to the MongoDB server
    client = pymongo.MongoClient( "mongodb+srv://ppay81755:P4vVeUjOViEbKuQv@cluster0.gi10wzb.mongodb.net/")

    db = client["chatbotesprit"]
    chat_sessions_collection = db["chat_sessions"]

    # Save the chat history for the selected bot
    chat_sessions_collection.update_one(
        {"bot_name": bot_name},
        {"$set": {"chat_history": chat_history}},
        upsert=True  # Create a new entry if not found
    )

    client.close()

# ... (previous code)
def main():

    st.set_page_config(page_title="Chatbot Esprit", page_icon=":robot_face:")
    print("MONGODB_URI:", os.environ.get("MONGODB_URI"))

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    if 'selected_bot' not in st.session_state:
        st.session_state.selected_bot = None

    st.write(css, unsafe_allow_html=True)

    st.header('Chatbot with multiple PDFs :books:')

    new_bot_name = st.text_input("Enter a new Chatbot Name:")
    if new_bot_name:
        handle_chatbot_creation(new_bot_name)

    display_chat_history_sidebar()

    new_chat_btn = st.sidebar.button("New Chat")
    if new_chat_btn:
        st.session_state.conversation = None
        st.session_state.chat_history = None

    user_question = st.text_input('Ask a question about your documents:')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your document and click on 'process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                store_pdf_text_chunks(st.session_state.selected_bot, text_chunks)
                save_pdf_chunks_to_db(st.session_state.selected_bot, text_chunks)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)

                if st.session_state.chat_history:
                    save_chat_history(st.session_state.selected_bot, st.session_state.chat_history)
                    save_chat_history_to_db(st.session_state.selected_bot, st.session_state.chat_history)

if __name__ == '__main__':
    main()