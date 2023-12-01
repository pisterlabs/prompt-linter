import os
import streamlit as st
from langchain import OpenAI, ConversationChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

os.environ['OPENAI_API_KEY'] = "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"

# Load and process the PDF files
loader = DirectoryLoader('./openrathaus_pdf/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

# Splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

st.title('OR-Chatbot')

# Initialize the list of previous responses in session state
if 'previous_responses' not in st.session_state:
    st.session_state.previous_responses = []

# Initialize the list of previous prompts in session state
if 'previous_prompts' not in st.session_state:
    st.session_state.previous_prompts = []

prompt = st.text_input("Stelle deine Frage!")

if prompt:
    llm_response = qa_chain(prompt)

    response = llm_response['result']
    sources = '\n'.join([source.metadata['source'] for source in llm_response["source_documents"]])

    # Add the prompt and response to the list of previous prompts and responses
    st.session_state.previous_prompts.append(prompt)
    st.session_state.previous_responses.append(response + "\n\nSources:\n\n" + sources)
    

# Display the list of previous prompts and responses in reverse order
if st.session_state.previous_prompts and st.session_state.previous_responses:
    for i in range(len(st.session_state.previous_prompts)-1, -1, -1):  # Reverse loop
        color = "#606060" if i % 2 == 0 else "#808080"
        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                    f"<b>Q:</b> {st.session_state.previous_prompts[i]}<br>"
                    f"<b>A:</b> {st.session_state.previous_responses[i]}"
                    f"</div>", unsafe_allow_html=True)
