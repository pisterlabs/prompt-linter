import streamlit as st
from math import ceil
import tempfile
import tiktoken
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from streamlit_chat import message

st.set_page_config(layout="wide",page_title="PrivateGPT",initial_sidebar_state="collapsed")
st.title('📚 :violet[Private]GPT')

st.sidebar.title('Settings')
openai_api_key = st.sidebar.text_input('Insert OpenAI API Key',type="password")
uploaded_file_pdf = st.sidebar.file_uploader("Upload Documents",type=["pdf","docx","doc"],accept_multiple_files=True)
# uploaded_file_pdf2 = st.sidebar.file_uploader("Upload PDF Files For 2nd Side Of Debate",type=["pdf"],accept_multiple_files=True)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def generate_response(input_text):
  llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
  st.info(llm(str(input_text)))

def generate_response2(input_text):
  out=pdf_qa({"query": str(input_text)})
  res=out['result']
  try:
    ref=''.join(["\n\n "+ "Source: \n" + i.metadata['source']+f"  page {i.metadata['page']+1}"+"\n" +"\n Content:"+"\n\n"+ i.page_content for i in out['source_documents']])
  except:
    ref=''.join(["\n\n "+ "Source: \n" + i.metadata['source']+"\n" +"\n Content:"+"\n\n"+ i.page_content for i in out['source_documents']])
  st.info(res+' \n \n '+"Reference: \n "+ref)
  

with st.form('my_form'):
  # Numtokens=0
  text = st.text_area('Enter text:', 'Upload Documents & Ask Me')
  # text=st.chat_input("Say something")
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if len(uploaded_file_pdf) != 0:
    docs=[]
    for uploadfile in uploaded_file_pdf:
      with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploadfile.getvalue())
        tmp_file_path = tmp_file.name
      pdf = PyPDFLoader(tmp_file_path)
      try:
        pages= pdf.load()
      except:
        loader = Docx2txtLoader(tmp_file_path)
        pages = loader.load()
      if(len(pages)<=120):
        chunk=10000
      else:
        chunk=min(ceil(61800*(len(pages)/1000)),61800)
      text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk, chunk_overlap = 0)
      texts = text_splitter.split_documents(pages)
      for i in texts:
        i.metadata['source']=uploadfile.name
      docs.extend(texts)
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":1})
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    res = " ".join([str(item.page_content) for item in docs])
    Numtokens = encoding.encode(res)
    st.sidebar.caption(f'Tokens:{len(Numtokens)}')
    pdf_qa= RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key,temperature=0.2,model_name='gpt-3.5-turbo-16k'), chain_type="stuff", retriever=retriever, return_source_documents=True)
  else:
    try:
      vectordb.delete_collection()
    except:
      pass
  if len(uploaded_file_pdf) != 0 and submitted and openai_api_key.startswith('sk-'):
      msg = st.toast('Gathering Information 🤓')
      generate_response2(text)
      st.toast("Got It ✅")
  if len(uploaded_file_pdf) == 0 and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
st.markdown("""
**:green[Instructions]**
1) Add API Key.
2) You Can Ask Any Questions Without Uploading Documents.
3) You Can Upload Documents From The Sidebar & Ask Related Questions.
4) :red[Experimental]: Arabic DOCX Documents Supported.
""")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
