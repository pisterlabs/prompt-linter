import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# load the document as before
loader = PyPDFLoader('./docs/dsa.pdf')
documents = loader.load()

# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./data'
)
vectordb.persist()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

# we can now execute queries against our Q&A chain
result = qa_chain({'query': 'What is the research paper about?'})
print(result['result'])