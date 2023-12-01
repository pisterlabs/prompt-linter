from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import os

class Pdf_reader:
  def __init__(self, pdf_file, openai_api_key):
    os.environ['OPENAI_API_KEY'] = openai_api_key
      
    self.reader = PdfReader(pdf_file) #library module init call
    raw_text = ' '
    for i, page in enumerate(self.reader.pages):
      text = page.extract_text()
      if text:
        raw_text += text
        
    text_splitter = CharacterTextSplitter(
      separator= '\n',
      chunk_size = 1000,
      chunk_overlap = 200,
      length_function = len)
    
    content = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(content, embeddings)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = docsearch.as_retriever(search_type= 'similarity', search_kwargs={'k':2})
    llm = OpenAI(model_name='text-davinci-003', temperature=0.5)
    self.qa =ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    
  def query(self, input_string: str):
    try:
      return self.qa({'question': input_string})['answer']
    except:
      return 'Something went wrong! Please try again.'
    