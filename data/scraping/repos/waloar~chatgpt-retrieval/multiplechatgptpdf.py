import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate

import constants
# REF https://blog.devgenius.io/chat-with-document-s-using-openai-chatgpt-api-and-text-embedding-6a0ce3dc8bc8
# ref https://levelup.gitconnected.com/langchain-for-multiple-pdf-files-87c966e0c032
# REF https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# no tiene buena performance para el caso.
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

documents = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        pdf_path = "./data/" + file
        loader = UnstructuredPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.data') or file.endswith('.doc'):
        doc_path = "./data/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./data/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# loader = PyPDFDirectoryLoader("data/")
# documents = loader.load()
print(len(documents))

#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
# Vectoriza los documentos
# vectordb = Chroma.from_documents(
#     documents, embeddings)
vectordb = Chroma.from_documents(
    documents, embeddings, persist_directory='persist')
vectordb.persist()

# Aplica el modelo de ChatGPT para conversacion
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Construye un templeta de prompt
template = """Utiliza todas las piezas del contexto para responder a al pregunta. Si no contestas la pregunta, simplemente di que no lo sabes,no trataes de crear una respuesta. Utiliza tres oraciones como maximo. Manten las respuestas concisas. Siempre di "Gracias por preguntar" al final de cada respuesta. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Utiliza la extraccion de datos conversacionales de pregunta y respuesta.

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
# chain=ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
# chain = RetrievalQA.from_chain_type(
#     llm,
#     return_source_documents=True,
#     retriever=vectordb.as_retriever(search_kwargs={"k": 1})
# )

# chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectordb.as_retriever(search_kwargs={"k": 1})
# )

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"query": query, "chat_history": chat_history})
    print(result['result'])

    chat_history.append((query, result['result']))
    query = None
