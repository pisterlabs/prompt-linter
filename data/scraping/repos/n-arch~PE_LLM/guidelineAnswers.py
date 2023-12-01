from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

openai_api_key = os.getenv("OPENAI_API_KEY", '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')

loader = DirectoryLoader('/Users/niklaskohl/Documents/GitHub/VeevaWorkflowLLM/guidelines', glob='**/*.txt')
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_documents(texts, embeddings)

llm = OpenAI(openai_api_key=openai_api_key,model='gpt-4')



#Gives back an answer about the guidlines
def ask_guideline(query):
    qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever(),
                                    return_source_documents=True)
    result = qa({"query": query})
    return result['result']#,result['source_documents']

