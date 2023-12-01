
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
# LLM
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

from langchain.document_loaders import DirectoryLoader
vectordb = Chroma(persist_directory="./data", embedding_function=GPT4AllEmbeddings())


# Prompt
template = """
Use the following pieces of context to answer the question at the end. 
You are an expert of the Operational Decision Manager (ODM).
If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
llm = Ollama(base_url="http://localhost:11434",
             model="mistral",
             verbose=False)

# QA chain
from langchain.chains import RetrievalQA

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(),chain_type="stuff",combine_docs_chain_kwargs={
                                     "prompt": QA_CHAIN_PROMPT
                                 },
                                                
                                                 memory=memory)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa_chain(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))