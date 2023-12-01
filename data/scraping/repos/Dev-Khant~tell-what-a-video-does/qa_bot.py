import logging

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Q&A")


class QA_Bot:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        self.llm = ChatOpenAI(
            temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=openai_key
        )
        self.agent = None

    def store_in_vectordb(self, explanation):
        document = Document(page_content=explanation)

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents([document])
        logger.info("Documents ready")

        vectordb = Chroma.from_documents(
            chunked_documents,
            embedding=OpenAIEmbeddings(openai_api_key=self.openai_key),
            persist_directory="./data",
        )
        vectordb.persist()
        logger.info("Documents inserted to vectordb")

        self.agent = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        )
        logger.info("Agent ready!!")

    def retrieve(self, query):
        result = self.agent({"query": query}, return_only_outputs=True)
        logger.info("Result ready!")
        return result["result"]
