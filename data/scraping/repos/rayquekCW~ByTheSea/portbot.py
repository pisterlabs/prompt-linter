import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
import os

# Chat UI title
st.header("PortBOT 🤖")
st.subheader('Helping You To Navigate Your PSA Career Journey! 🛳️')

# # File uploader in the sidebar on the left
# with st.sidebar:
#     openai_api_key = st.text_input("Login Key", type="password")
# if not # The `openai_api_key` variable is used to store the OpenAI API key, which is required to
# access the OpenAI GPT-3 language model. This key is used to authenticate and authorize the
# requests made to the OpenAI API.
# openai_api_key:
#     st.info("Please add your Login Key to continue.")
#     st.stop()

# # Set OPENAI_API_KEY as an environment variable
# os.environ["OPENAI_API_KEY"] == openai_api_key

llm = ChatOpenAI(temperature=0,max_tokens=1000, model_name="gpt-3.5-turbo",streaming=True)
        
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your resume in .docx format", accept_multiple_files=True, type=None)
    st.info("Please refresh the browser to reset the session if you decided to upload a new resume or start a new conversation", icon="🚨")
    
# Check if files are uploaded
if uploaded_files:

    # Print the number of files to console
    print(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded PDF files
        documents = []
        # Add knowledge base file to documents for training
        base_file_path = os.path.join(os.getcwd(), "external-knowledge-base-file.docx")

        # Define the path of your default knowledge base file
        default_knowledge_base_path = "portbot_app/external-knowledge-base-file.docx"

        # Read the content of the default knowledge base file
        with open(default_knowledge_base_path, "rb") as f:
            default_knowledge_base_content = f.read()

        # Save the knowledge base file to disk
        with open(base_file_path, "wb") as f:
            f.write(default_knowledge_base_content)

        # Use UnstructuredFileLoader to load the knowledge base file
        base_file_loader = UnstructuredFileLoader(base_file_path)
        base_loaded_documents = base_file_loader.load()
        
        # Extend the main documents list with the loaded documents
        documents.extend(base_loaded_documents)

        for uploaded_file in uploaded_files:
            # Get the full file path of the uploaded file
            file_path = os.path.join(os.getcwd(), uploaded_file.name)

            # Save the uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Use UnstructuredFileLoader to load the PDF file
            loader = UnstructuredFileLoader(file_path)
            loaded_documents = loader.load()
            print(f"Number of files loaded: {len(loaded_documents)}")

            # Extend the main documents list with the loaded documents
            documents.extend(loaded_documents)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

        # Print the number of total chunks to console
        print(f"Number of total chunks: {len(document_chunks)}")

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Initialize Langchain's QA Chain with the vectorstore
    qa = ConversationalRetrievalChain.from_llm(llm,vectorstore.as_retriever())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check if the introductory message has already been sent
    if "intro_sent" not in st.session_state:
        # Custom bot name
        custom_name = "PortBOT"

        # Display custom introduction message
        custom_intro = f"Hi there! 👋 I'm {custom_name}, your personalized recruitment assistant. Whether you're exploring job opportunities, need assistance with your application, or have questions about PSA, I'm here to help. Feel free to ask me anything about our open positions, company culture, or application process. Let's get started on finding the perfect role for you! 🚀"

        # Write custom intro to assistant messages
        st.session_state.messages.append({"role": "assistant", "content": custom_intro})

        # Mark the introductory message as sent
        st.session_state["intro_sent"] = True

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the assistant using the latest chat history
        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your resume to begin.")
