import sys

sys.path.append(r"../")
from langchain.llms import OpenAI

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)
# chat = ChatOpenAI(
#     temperature=0,
#     verbose=True
# )

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your data
loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
data = loader.load()
print(f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} characters in your document')

# Chunk your data up into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print(f'Now you have {len(texts)} documents')

# Create embeddings of your documents to get ready for semantic search
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./vector_store")
# vector_store.persist()
docsearch = Chroma(persist_directory="./vector_store", embedding_function=embeddings)
query = "What are examples of good data science teams?"
docs = docsearch.similarity_search(query, include_metadata=True)

# # Query those docs to get your answer back
# from langchain.chains.question_answering import load_qa_chain
# chain = load_qa_chain(llm, chain_type="stuff")
# query = "What is the collect stage of data maturity?"
# docs = docsearch.similarity_search(query, include_metadata=True)
# chain.run(input_documents=docs, question=query)

# Query those docs to get your answer back
from langchain.chains import RetrievalQA

retriever = docsearch.as_retriever()
# 创建一个链并用它来回答问题！
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
# print(qa.input_keys, qa.output_keys)
query = "出差申请单修改"
print(qa.run(query=query))
