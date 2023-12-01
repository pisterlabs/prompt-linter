import streamlit as st
from streamlit import session_state
from dotenv import load_dotenv
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import warnings

warnings.filterwarnings("ignore")

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.postimg.cc/P5phFqqh/bot.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.postimg.cc/MHMdJgfc/human.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

button_css = '''
<style>
    div[class="row-widget stButton"] > button {
        width: 20%;
        height: 60px;
        border-radius: 5px;
        font-size: 20px;
    } 
</style>
'''

css = '''
<style>
    body {
        font-family: 'Open Sans', sans-serif;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.15);
        border-radius: 1rem;
    }
    .chat-message:hover {
        transform: scale(1.05);
    }
    .chat-message.user {
        background-color: #2b313e;
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
    @keyframes slideIn {
        0% {
            transform: translateX(-50px);
            opacity: 0;
        }
        100% {
            transform: translateX(0);
            opacity: 1;
        }
    }
    .chat-message {
        animation: slideIn 0.5s ease-out;
    }
</style>
'''

text_input_css = """  
<style>
    .element-container:has(> div[class="row-widget stTextInput"]) {
        position: fixed !important;
        bottom: 0 !important;
        left: 0;
        right: 0;
    }
</style>  
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
   
def get_text_chunks(raw_text):
    # To ensure the model gets the meaning of the full sentence
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
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

    load_dotenv() # load environment variables from a .env file

    press_button = False
    press_enter = False

    st.set_page_config(page_title="📚 PDFPal", page_icon="📚")
    # https://i.postimg.cc/P5phFqqh/bot.png
    st.write(css, unsafe_allow_html=True)
    st.write(button_css, unsafe_allow_html=True)
    st.markdown(text_input_css, unsafe_allow_html=True) 

    st.title("📚 PDFPal")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

    if "conversation" not in st.session_state: st.session_state.conversation = None

    if "chat_history" not in st.session_state: st.session_state.chat_history = None

    if "buffer" not in st.session_state: st.session_state.buffer = False
    
    if "text_detector" not in st.session_state: st.session_state.text_detector = False

    st.header("Ask anything. Know everything. :brain:")
    st.write("""---""") 

    with open("./files/AWS_certification_paths.pdf", "rb") as file:
        st.download_button(
            label="Download PDF",
            data=file,
            file_name="AWS_certification_paths.pdf",
        )
    st.markdown('''<p style="color:red;">PDF-less? We've got you covered. Snag an awesome sample with one click below!</p>''', unsafe_allow_html=True)
    st.subheader('Your PDFs')

    pdf_docs = st.file_uploader("Drop files, click Process, watch our AI work its magic!", accept_multiple_files=True)

    if pdf_docs is not None and 1 <= len(pdf_docs) <= 3:
        if pdf_docs:
            press_button = st.button("Process")

        if press_button:
            with st.spinner("Processing your documents..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store
                vectorstore = get_vectorstore(text_chunks)

                # Create Conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.session_state.buffer = True
                
        st.write("""---""") 
        if st.session_state.conversation:   
            user_question = st.text_input("Delve into the insights within your PDFs:", placeholder="Ask me anything...", key="user_input_before")
            if user_question:
                st.session_state.text_detector = True
            press_enter = st.button("Enter!")
            
        if press_enter and st.session_state.text_detector:
            st.session_state.buffer = False
            with st.spinner("Untangling neurons..."):
                handle_userinput(user_question)  
        if press_enter and not st.session_state.text_detector:
            st.warning("Please enter a prompt.")
    else:
        st.warning("*** Maximum of three PDFs can be uploaded simultaneously ***")
        st.warning("*** For PDF reprocessing or adding additional files, simply hit that refresh button! ***")

if __name__ == '__main__': # define code that should only run when the script is executed directly
    main()