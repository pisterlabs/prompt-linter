from langchain.retrievers.web_research import WebResearchRetriever
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain


# Vectorstore
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
)

# LLM
model = ChatOpenAI(temperature=0)

# Search
os.environ["GOOGLE_CSE_ID"] = "xxx"
os.environ["GOOGLE_API_KEY"] = "xxx"
search = GoogleSearchAPIWrapper()
# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=model,
    search=search,
)
user_input = "How do LLM Powered Autonomous Agents work?"
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_research_retriever
)
result = qa_chain({"question": user_input})
result
