import dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load OPENAI_API_KEY
dotenv.load_dotenv()

# 1. Load
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 3. Store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 4. Retrieve
retriever = vectorstore.as_retriever()

# 5.+6. Generate + Chat
llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

print(qa("How do agents use Task decomposition?"))
print(qa("What are the various ways to implement memory to support it?"))
