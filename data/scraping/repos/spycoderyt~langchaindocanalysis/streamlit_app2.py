# Import os to set API key
import os
import langchain
import PyPDF2
import io
from streamlit_chat import message
import openai
# Import OpenAI as main LLM service
from langchain.llms import OpenAI 
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st
from langchain.memory import ConversationBufferMemory

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Pinecone, Chroma
from langchain.document_loaders import GoogleDriveLoader
from langchain.chains import ConversationalRetrievalChain

from langchain.chains import ConversationChain
import tempfile
from PyPDF2 import PdfWriter
from langchain.document_loaders import TextLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, OpenAI, LLMChain
from urllib.parse import urlparse, parse_qs
prev_file_upload = None
prev_url = None

def create_empty_pdf():
    pdf_writer = PdfWriter() 

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        pdf_writer.write(temp)
        temp_path = temp.name

    return temp_path
def make_qa():
    st.session_state.qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(model_name ="gpt-3.5-turbo-16k",temperature=0), retriever=st.session_state.store.as_retriever(), memory=st.session_state.memory,verbose = True)
def load_data(uploaded_files):
    # Initialize an empty Chroma store
    st.session_state.file_uploaded = True
    st.spinner(text="Received document {uploaded_file.name}...")    
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        # now temp_file_path is the path of the temporary file, which you can pass to PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()
        # Add documents to the existing Chroma store
        st.write(f"Processing file: {uploaded_file.name}")
        if 'store' not in st.session_state:
            st.session_state.store = Chroma.from_documents(docs, st.session_state.embeddings)
        else:
            st.session_state.store.add_documents(docs)
        st.write(f"Document {uploaded_file.name} added!")
def extract_id(url):
    parsed_url = urlparse(url)
    if 'google.com' in parsed_url.netloc:
        if 'd' in parsed_url.path.split('/'):
            doc_id = parsed_url.path.split('/')[parsed_url.path.split('/').index('d')+1]
            return doc_id
        elif 'id' in parse_qs(parsed_url.query):
            return parse_qs(parsed_url.query)['id'][0]
    return None
def load_google_docs(urls):
    st.session_state.file_uploaded = True
    # Split the input urls by newline to get a list of URLs
    urls = urls.split(',')
    for url in urls:
        document_id = []
        document_id.append(extract_id(url))
        st.write(f"Processing document with id: {document_id[0]}")
        loader = GoogleDriveLoader(
            document_ids=document_id,
            credentials_path='credentials.json',
            token_path='token.json'
        )
        docs = loader.load()
        if 'store' not in st.session_state:
            st.session_state.store = Chroma.from_documents(docs,st.session_state.embeddings)
        else:
            st.session_state.store.add_documents(docs)
            st.write("Document added!")

def process_openai_key(OPENAI_API_KEY):
    st.session_state.thekey = OPENAI_API_KEY
    st.session_state.key_entered = True
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    st.session_state.embeddings = OpenAIEmbeddings()
def init():
    if 'past_queries' not in st.session_state:
        st.session_state.past_queries = []
    if 'past_answers' not in st.session_state:
        st.session_state.past_answers = []
    st.session_state.key_entered = False
    st.session_state.file_uploaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.cnt = 1
    if 'prev_file_upload' not in st.session_state:
        st.session_state.prev_file_upload = None
    if 'prev_url' not in st.session_state:
        st.session_state.prev_url = None
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
def process_key_entered(prompt):
    make_qa()
    response=""
    if(st.session_state.file_uploaded):
        res = st.session_state.qa({"question":prompt})
        response = res["answer"]
        with st.expander('Document Similarity Search'):
            search = st.session_state.store.similarity_search_with_score(prompt) 
            st.write(search[0][0].page_content) 
    else:
        return
    
    st.session_state.chat_history.append("This is the user's query number {st.session_state.cnt}")
    st.session_state.chat_history.append(prompt)
    st.session_state.chat_history.append("This was your response for query number {st.session_state.cnt}")
    st.session_state.chat_history.append(response)
    # message(response)
    st.session_state.past_queries.append(prompt)
    st.session_state.past_answers.append(response)
    st.session_state.cnt+=1
    output_text = f"Prompt: {prompt}\nResponse: {response}"
    b = io.BytesIO()
    b.write(output_text.encode())
    b.seek(0)
    st.download_button("Download Prompt and Response", b, file_name='prompt_and_response.txt', mime='text/plain')
def main(): 
    init()
    st.title('🦜🔗 Langchain Document Analytics')
    st.markdown('[Documentation/Github](https://github.com/spycoderyt/langchaindocanalysis)')
    if st.button('reset the bot'):
        make_qa()

    c1,c2 = st.columns(2)
    with c1:
        prompt = st.text_area('Input your prompt here')
        OPENAI_API_KEY = st.text_input('Please enter your OpenAI API Key!',type="password")
    #receive api key
    if(len(OPENAI_API_KEY)):
        process_openai_key(OPENAI_API_KEY)
    else:
        st.session_state.key_entered = False
    with c2:
        file_upload = st.file_uploader("Please upload a .pdf file!", type="pdf", accept_multiple_files=True, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        url = st.text_area('Input a link to a google drive file to be scraped! (for multiple files, separate with commas). For google drive API setup refer to documentation.')
    
    if len(file_upload):
        st.session_state.file_uploaded = True
        if file_upload != st.session_state.prev_file_upload:
            st.session_state.data_loaded = False
            st.session_state.prev_file_upload = file_upload
        if not st.session_state.data_loaded:
            load_data(file_upload)
            st.session_state.data_loaded = True

    if url:
        if url != st.session_state.prev_url:
            st.session_state.data_loaded = False
            st.session_state.prev_url = url
        if not st.session_state.data_loaded:
            load_google_docs(url)
            st.session_state.data_loaded = True
    if prompt:
        if(st.session_state.key_entered):
            process_key_entered(prompt)
        else:
            st.session_state.past_answers.append("plz enter valid openai api key :)")
    # if len(st.session_state.past_queries) > 0:
    #     st.subheader('Past Queries and Answers')
    #     for i, (query, answer) in enumerate(zip(st.session_state.past_queries, st.session_state.past_answers)):
    #         st.write(f'**Query {i+1}:** {query}')
    #         st.write(f'**Answer {i+1}:** {answer}')
    #         st.write('---')
    if len(st.session_state.past_queries) > 0:
        st.subheader('Past Queries and Answers')
        for i, (query, answer) in enumerate(zip(st.session_state.past_queries[::-1], st.session_state.past_answers[::-1])):
            st.write(f'**Query {len(st.session_state.past_queries)-i}:**')
            message(query,is_user=True,key=2*(len(st.session_state.past_queries)-i-1))
            message(answer,key=2*(len(st.session_state.past_queries)-i-1)+1)
            st.write('---')


if __name__ == "__main__":
    main()