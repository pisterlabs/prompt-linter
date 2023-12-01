# Import necessary modules
import pandas as pd
import streamlit as st 
from PIL import Image
from PyPDF2 import PdfReader

from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

home_privacy = "We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private."

# Page configuration for Simple PDF App
st.set_page_config(
    page_title="Document Q&A with AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
st.sidebar.subheader("Setup")
OPENAI_API_KEY = st.sidebar.text_input("Enter Your OpenAI API Key:", type="password")
st.sidebar.markdown("Get your OpenAI API key [here](https://platform.openai.com/account/api-keys)")
st.sidebar.divider()
st.sidebar.subheader("Model Selection")
llm_model_options = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4']  # Add more models if available
model_select = st.sidebar.selectbox('Select LLM Model:', llm_model_options, index=0)
st.sidebar.markdown("""\n""")
temperature_input = st.sidebar.slider('Set AI Randomness / Determinism:', min_value=0.0, max_value=1.0, value=0.5)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("Clear conversation history")


with st.sidebar:
    st.divider()
    st.subheader("Considerations:", anchor=False)
    st.info(
        """
        - Currently only supports PDFs. Include support for .doc, .docx, .csv & .xls files 

        """)

    st.subheader("Updates Required:", anchor=False)
    st.warning("""
        1. Support for multiple PDFs.
        
        2. Use Langchain PDF loader and higher quality vector store for document parsing + reduce inefficient handling.
        
        3. Improve contextual question-answering by developing Prompt Templates - Tendency to hallucinate.
    
        """
        )

    st.divider()

with st.sidebar:
    st.subheader("👨‍💻 Author: **Yaksh Birla**", anchor=False)
    
    st.subheader("🔗 Contact / Connect:", anchor=False)
    st.markdown(
        """
        - [Email](mailto:yb.codes@gmail.com)
        - [LinkedIn](https://www.linkedin.com/in/yakshb/)
        - [Github Profile](https://github.com/yakshb)
        - [Medium](https://medium.com/@yakshb)
        """
    )

    st.divider()
    st.write("Made with 🦜️🔗 Langchain and OpenAI LLMs")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

st.markdown(f"""## AI-Assisted Document Analysis 📑 <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)
st.write("_A tool built for AI-Powered Research Assistance or Querying Documents for Quick Information Retrieval_")

with st.expander("❔How does the report analysis work?"):
    st.info("""
    These processes are powered by robust and sophisticated technologies like OpenAI’s Large Language Models, Sentence Transformer, FAISS, and Streamlit, ensuring a reliable and user-friendly experience for users to gain quick insights from their documents.

    1. **Document Upload and Processing**: The tool reads and extracts text from these documents, creating a foundational base of information. During this phase, the documents are also processed into manageable pieces to prepare them for subsequent analysis and querying.
    
    2. **Data Transformation and Indexing**: HuggingFace Sentence Transformers convert textual data into numerical vectors. Post-transformation, the data is organized and indexed in a vector database using Meta's FAISS, which is renowned for its efficient search capabilities.
    
    3. **Conversational AI**: Using OpenAI's ChatOpenAI model to generate responses, the system retrieves answers based on the information extracted from the uploaded documents while maintaining contextual accuracy.
    
    4. **Query Handling and Response Generation**: Each user query is meticulously managed and processed within the tool. The tool ensures a smooth interaction and generates accurate responses based on the ongoing conversation and the available data.
    
    The overarching objective is to enable users to query lengthy documents or reports to expedite comprehensive research.

    """, icon="ℹ️")

with st.expander("⚠️ Privacy and Terms of Use"):
    st.info("""
        **Privacy**: We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private.

        **Terms of Use**: By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. 
        It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. 
        The service may collect user dialogue data for future research. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

        **License**: The service is a research preview intended for non-commercial use only, subject to the model [License](https://huggingface.co/docs/hub/sentence-transformers) HuggingFace embedding models, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI and Privacy Practices of Langchain. Please contact us if you find any violations.
        """, icon="ℹ️")

# Extracts and concatenates text from a list of PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
        except (PdfReader.PdfReadError, PyPDF2.utils.PdfReadError) as e:
            print(f"Failed to read {pdf}: {e}")
            continue  # skip to next pdf document in case of read error

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # checking if page_text is not None or empty string
                text += page_text
            else:
                print(f"Failed to extract text from a page in {pdf}")

    return text

# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=temperature_input, model_name=model_select),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h : h,
        memory=memory
    )
    return conversation_chain


# Upload file to Streamlit app for querying
user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("Processing"):
            # Get PDF Text
            raw_text = get_pdf_text(user_uploads)
            # st.write(raw_text)

            # Retrieve chunks from text
            text_chunks = get_text_chunks(raw_text)
            ## st.write(text_chunks)  

            # Create FAISS Vector Store of PDF Docs
            vectorstore = get_vectorstore(text_chunks)

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
            


# Initialize chat history in session state for Document Analysis (doc) if not present
if 'doc_messages' not in st.session_state or clear_history:
    # Start with first message from assistant
    st.session_state['doc_messages'] = [{"role": "assistant", "content": "Query your documents"}]
    st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

# Display previous chat messages
for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# If user provides input, process it
if user_query := st.chat_input("Enter your query here"):
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    # Add user's message to chat history
    st.session_state['doc_messages'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Generating response..."):
        # Check if the conversation chain is initialized
        if 'conversation' in st.session_state:
            st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            # Process the user's message using the conversation chain
            result = st.session_state.conversation({
                "question": user_query, 
                "chat_history": st.session_state['chat_history']})
            response = result["answer"]
            # Append the user's question and AI's answer to chat_history
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": response
            })
        else:
            response = "Please upload a document first to initialize the conversation chain."
        
        # Display AI's response in chat format
        with st.chat_message("assistant"):
            st.write(response)
        # Add AI's response to doc_messages for displaying in UI
        st.session_state['doc_messages'].append({"role": "assistant", "content": response})

