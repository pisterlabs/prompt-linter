from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import StdOutCallbackHandler

loader = PyPDFLoader("./data/syllabus.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150
)

data = loader.load_and_split(text_splitter=text_splitter)
print(data[0].page_content)

embeddings = OpenAIEmbeddings(show_progress_bar=True)
index = FAISS.from_documents(data, embeddings)

retriever = index.as_retriever()
retriever.search_kwargs['fetch_k'] = 20
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

llm = ChatOpenAI()

chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,
    verbose=True
)

handler = StdOutCallbackHandler()

chain.run(
    'What are the prerequisites for the course?',
    callbacks=[handler]
)

