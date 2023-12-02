import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import urllib.request
import urllib.parse
import os, json

os.environ['OPENAI_API_KEY'] = st.secrets['OPEN_AI_KEY'] 
chat = ChatOpenAI(model_name="gpt-3.5-turbo")


try:
    os.mkdir('book')
    os.mkdir('txt')
except:
    pass

url = "https://www.impromptubook.com/wp-content/uploads/2023/03/impromptu-rh.pdf"

def llm(prompt):
    res = chat([HumanMessage(content=prompt)])
    return res.dict()['content']

def get_book(url):
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    # Extract the path from the parsed URL
    path = parsed_url.path
    # Use os.path to get the file name from the path
    file_name = os.path.basename(path)
    # Unquote the file name to convert %20 to spaces and handle other special characters
    pdf = urllib.parse.unquote(file_name)
    urllib.request.urlretrieve(url, os.path.join('book',pdf))
    return pdf

@st.cache_resource
def index_doc(book):
    # loader = DirectoryLoader('book')
    # documents = loader.load()
    loader = PyPDFLoader(book)
    documents = loader.load_and_split()
    # st.write(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    # st.write(len(texts))

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def change_prompt(prompt):
    st.session_state.prompt = prompt

def get_sources(res):
    for r in res['source_documents']:
        # st.write(r)
        st.write('**Page** ',r.metadata['page'])
        st.write(r.page_content)
        st.write('---')

def follow_up(res):
    follow_up_prompt = f'''
    Provide a list of 3 follow-up questions to the initial query and the result associated. 
    Here is the initial query:
    {res['query']}
    Here is the result associated:
    {res['result']}
    Format the output as a JSON file with a list of the 3 follow-up questions in a field called follow-up
    '''
    q2 = llm(follow_up_prompt)
    jason = json.loads(q2)
    # st.write(jason)

    for element in jason['follow-up']:
        st.button(element, on_click=change_prompt, args=(element,))

st.title("🤖 Question Answering on Doc")

if 'prompt' not in st.session_state:
    st.session_state.prompt = "How can AI be used for education?"

src = st.radio('Doc source',('Book demo','Upload'))

if src == 'Book demo':
    st.markdown(f"Source: 📖 [Impromptu]({url})")
    if not os.path.exists("book/impromptu-rh.pdf"):
        # if st.button('Get Book'):
        pdf = get_book(url)
            # st.write("downloading ... ",pdf)
    else:
        pdf = "impromptu-rh.pdf"
else:
    uploaded_file = st.file_uploader('Upload','pdf')
    if uploaded_file is not None:
        # st.write(uploaded_file)
        pdf = uploaded_file.name
        with open(os.path.join('book', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

if 'pdf' in locals():
    # st.write(pdf)
    i = st.checkbox('Index doc')
    if i:
        # pdf = "impromptu-rh.pdf"
        qa = index_doc("book/"+pdf)

    query = st.text_area('Query',value=st.session_state.prompt)
    if st.button('Answer') & i:
        res = qa({"query": query})
        # st.write(res)
        st.write(res['result'])
        # st.write("### Sources:")
        with st.expander('Sources:'):
            # st.write(res.get_formatted_sources().replace('>',''))
            get_sources(res)
        st.write('Follow-up questions:')
        follow_up(res)