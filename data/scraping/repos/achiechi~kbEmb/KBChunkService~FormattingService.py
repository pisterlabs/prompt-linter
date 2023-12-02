from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import tiktoken
import KBChunkService.AzureOpenAI as OpenAIService
from typing import List

tokenizer = tiktoken.get_encoding('cl100k_base')

# Format the content and create chunks basd on token size
def text_to_docs(page_content:str) -> List[Document]:
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
    chunks = text_splitter.split_text(page_content)
    for chunk in chunks:
        document = Document(
            page_content=chunk
        )
        documents.append(document)
    return documents

#Generate Embeddings and store in dataframe
def start_process(content, f_name):
    resultList = []
    for idx, page in enumerate(content):
      Filename = f_name+"-"+str(idx)
      Content = page
      Embeddings = OpenAIService.get_embeddings(page)
      myList = [Filename, Content, Embeddings]
      resultList.append(myList)
    df = pd.DataFrame(resultList, columns=['Filename','Content', 'Embeddings'])
    return df

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)
