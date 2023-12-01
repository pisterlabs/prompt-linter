# Import all necessary modules

from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import tiktoken
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

# Note: If you have a pdf file, there are a lot of free tools to convert pdf to txt. Convert to txt first.

# Set .txt path
path_f = 'test.txt'

# Load and process the text
loader = TextLoader(path_f)
documents = loader.load()

# Split the loaded txt to chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# To check if our chunks are loaded
print(type(texts))
print(texts[0])
print("[+]=====[+]")
print("[+]Done --> Splitting txt file to chunks")

# Check amount of token of the loaded txt file
with open(path_f, 'r') as file:
    content = file.read()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
print("[+]=====[+]")
numtoken = num_tokens_from_string(content, "cl100k_base")
print("Number Of Token: " + str(numtoken))

# Embedding the Chunks, and saving to a vectorstore
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5", max_length=512)

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    path="./local_qdrant1",
    collection_name="db2",
)
print("[+]=====[+]")
print("[+]Done  --> Embedding the Chunks, and saving to a vectorstore")

# Performing similarity search on the embedding
question = input("Enter Any Question Related to the documents: ")
docs = qdrant.similarity_search(question)

print(docs[0].page_content)
len(docs)

# Preparing our chat model to make the similarity search result more human-like
template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. <</SYS>>
{context}
Question: {question}
Helpful Answer:[/INST]"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
print("[+]=====[+]")
print("[+]Done --> Preparing our chat model to make the similarity search result more human-like")

# Choosing the free llama 7B for the task (Excellent)
chat_model = ChatOllama(
    model="llama2",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
print("[+]=====[+]")
print("[+]Done --> Choosing the free llama 7B for the task (Excellent)")

# Adding a retrieval QA
qa_chain = RetrievalQA.from_chain_type(
    chat_model,
    retriever=qdrant.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
print("[+]=====[+]")
print("[+]Done --> Adding a retrieval QA")
print("[+]Go ahead and use the bot to answer any question from your documents (use ctrl + c to end the bot chat)")

# We are now ready, and good to go to ask question based on our document, and get human like answer
while True:
    try:
        question = input("Enter Prompt: ")
        print("Question: " + question)
        print("Answer ->>> ")
        result = qa_chain({"query": question})
        print("\n")
    except KeyboardInterrupt:
        print("[+]==[+]")
        print("Bye! : Ending the Bot Chat")
        break
