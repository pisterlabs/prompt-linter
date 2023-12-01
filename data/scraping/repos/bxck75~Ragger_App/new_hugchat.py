import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
from credits import HUGGINGFACE_EMAIL,HUGGINGFACE_PASS,HUGGINGFACE_TOKEN
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import FolderLoader
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': False})


# Create an instance of HuggingFaceEmbeddings:

# App title
st.set_page_config(page_title="🤗💬 HugChat")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬 HugChat')
    hf_email = HUGGINGFACE_EMAIL
    hf_pass = HUGGINGFACE_PASS

    st.success('Correct email and pass! Proceed to entering your prompt message!', icon='👉')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
    # File upload
    uploaded_file = st.file_uploader('Upload an article', type='txt')
    files_folder = st.text_input('folder to files')



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    #sign = Login(email, passwd)
    #cookies = sign.login()
    # Create ChatBot                        
    #bot = hugchat.ChatBot(cookies=cookies.get_dict())
    from HuggingChatAPI import HuggingChat
    llm = HuggingChat(email = email , psw = passwd) #for start new chat
    from langchain.document_loaders import FolderLoader


   
    if files_folder is not None:
        #• Specify the folder path containing the files you want to load:

        folder_path = file


        #• Create a FolderLoader object to load the documents from the folder:

        loader = FolderLoader(folder_path)
        documents = loader.load()

    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=16,
            length_function=len,
            is_separator_regex=False,
        )
        # Split
        texts = text_splitter.create_documents(documents)

        # Create a vectorstore from documents
        db = FAISS.from_documents(texts, hf_embeddings)
        # Create retriever interface
        retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
        #retriever = VectorStoreRetriever(vectorstore=db)
        qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)
        return qa.run(prompt_input)





# User-provided prompt
#if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
if prompt := st.chat_input('Enter your question:', disabled=not (hf_email and hf_pass) ):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, hf_email, hf_pass) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)