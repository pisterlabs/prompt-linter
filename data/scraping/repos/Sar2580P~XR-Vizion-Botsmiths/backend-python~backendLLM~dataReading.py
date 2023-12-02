from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader, UnstructuredEPubLoader
from langchain.schema import Document
from typing import List
from langchain.text_splitter import SpacyTextSplitter, RecursiveCharacterTextSplitter
from backendLLM.chains import *


class DataReading:
  def __init__(self , paths):
    self.sources = paths
    self.docs : List[Document] =[]

  def __read_pdf__(self , path)-> List[Document]:
    loader = PyPDFLoader(file_path=path)
    pdf_pages =  loader.load()
    for doc in pdf_pages:
      doc.metadata['source'] = path
    return pdf_pages
  
  def __read_txt__(self , path)-> List[Document]:
    loader = TextLoader(path)
    return loader.load()

  def __read_web_page__(self, url)-> List[Document]:
    loader = WebBaseLoader(web_path=url ,verify_ssl=True)    # To bypass SSL verification errors during fetching, set verify_ssl=False
    return loader.load()
  
  def __read_docx__(self , path)-> List[Document]:
    loader = Docx2txtLoader(path)
    return loader.load()
  
  def __read_epub__(self , path)-> List[Document]:
    loader = UnstructuredEPubLoader(path)
    return loader.load()
  
  def read_all(self):
    for source in self.sources:
      if source.endswith('.pdf'):
        self.docs.append(self.__read_pdf__(source))
      elif source.endswith('.txt'):
        self.docs.append(self.__read_txt__(source))
      elif source.endswith('.docx') or source.endswith('.doc'):
        self.docs.append(self.__read_docx__(source))
      elif source.endswith('.epub'):
        self.docs.append(self.__read_epub__(source))
      else:
        try:
          self.docs.append(self.__read_web_page__(source))
        except :
          print('Document format not supported ---> {}'.format(source))
    return self.docs
  
#_________________________________________________________________________________________
class DataSplitting:
  def __init__(self , docs: List[Document], enrich_metadata:bool = False):
    self.docs = docs
    self.pages:List[Document] = []
    self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len,)
    self.enrich_metadata = enrich_metadata
    
  def __get_metadata__(self, text)->dict:
    return title_chain.run(text) , summary_chain.run(text)
  
  def split(self):
    for pages in self.docs:
      for page in pages:
        meta = page.metadata
        texts:List[str] = self.text_splitter.split_text(page.page_content)

        for text in texts:
          if self.enrich_metadata:
            title, summary = self.__get_metadata__(text) 

            meta['summary'] = summary
            meta['title'] = title
          self.pages.append(Document(page_content=text , metadata=meta))
    return self.pages
  
#_________________________________________________________________________________________
    
  



