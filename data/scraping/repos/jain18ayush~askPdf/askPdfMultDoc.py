from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader

#this is the process for doing persistent storage: https://colab.research.google.com/drive/1gyGZn_LZNrYXYXa-pltFExbptIe7DAPe?usp=sharing#scrollTo=Q_eTIZwf4Dk2

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
persist_directory = './db'


# loads a pdf into a document object
loader = DirectoryLoader('/Users/ayushjain/OneSpace/Untitled/pdf/', glob="./*.pdf", loader_cls=PyPDFLoader)
# loader = PyPDFLoader("/Users/ayushjain/Downloads/case 1.pdf")
data = loader.load()

# the other option is the character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, length_function=len,)
# the splits are fed into the embedder and then fed into the vector store
all_splits = text_splitter.split_documents(data)

# # creates the vector store using the OPENAI Embedding
# vectorstore = Chroma.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)

# vectorstore.persist()

# #how to retrieve data at a later point 
# vectorstore = None 
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())


#add a document at a later point 
# loader = PyPDFLoader("/Users/ayushjain/Downloads/Influence is Power_ The Creation of Cultural Capital.pdf")
# data = loader.load()

# # the other option is the character text splitter
# # the splits are fed into the embedder and then fed into the vector store
# all_splits = text_splitter.split_documents(data)

# vectorstore.add_documents(documents=all_splits)
# vectorstore.persist()

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),  # the retriever is the vectorStore
                                                  llm=llm)  # it asks the query in numerous ways to gpt to create the most optimal + consistent answer

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# retriever from llm is much slower so possibly stick to normal retriever for now
retriever = vectorstore.as_retriever()
chat = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever_from_llm, memory=memory)

# print(chat.question_generator.prompt)

# result = chat(
#     {"question": "What are the possible failure points when submitting a motion?"})
# docs: https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html#langchain.vectorstores.chroma.Chroma.similarity_search_with_relevance_scores

#provides teh possible sources
#vectorstore.similarity_search_with_relevance_scores(result["answer"])

chat_history = []
# result = chat({"question": "What is cultural capital?"})
qInput = input("Provide a Question: ") 
result = chat({"question": qInput})
print(result["answer"])

qInput = input("Provide a Question: ") 
result = chat({"question": qInput})
print(result)
# print(vectorstore.similarity_search_with_relevance_scores(result["answer"], 5, score_threshold=0.7))
# do a similarity check on the vectorstore
