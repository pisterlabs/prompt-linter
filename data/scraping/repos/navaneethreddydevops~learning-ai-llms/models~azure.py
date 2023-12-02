from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

loader = UnstructuredFileLoader('../data/Cloud-Native-Devops-with-Kubernetes-full-book.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
doc_search = Chroma.from_documents(texts, embeddings)
chain = RetrievalQA.from_chain_type(llm=AzureOpenAI(model_kwargs={'engine': 'text-davinci-002'}), chain_type='stuff',
                                    retriever=doc_search.as_retriever())

query = 'what are the kubernetes pods'

chain.run(query)
