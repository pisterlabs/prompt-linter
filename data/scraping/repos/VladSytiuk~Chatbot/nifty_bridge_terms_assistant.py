from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFium2Loader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from openai.error import InvalidRequestError

from app.errors.app_errors import TokenLimitExceededError
from app.services.base import BaseService


TEMPLATE = """The chat bot should introduce itself as "NiftyBridge AI assistant".
Chat should not answer questions that are not related to the
the Nifty Bridge program.
If the chat has no answer, it should say specifically: "I don't know
please contact support by email support@nifty-bridge.com".
Chat should look for an answer from vectorstore documents.
{context}
Question: {question}
Helpful Answer:"""


class NiftyBridgeTermsAssistantService(BaseService):
    QA_CHAIN_PROMPT = PromptTemplate.from_template(TEMPLATE)

    async def process_question(self, question: str) -> str:
        raw_document = await self._load_document()
        chunked_documents = await self._split_document(raw_document)
        vector_store = await self._store_documents(chunked_documents)
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        )
        try:
            answer = qa_chain({"query": question})
        except InvalidRequestError:
            raise TokenLimitExceededError()
        return answer["result"]

    @staticmethod
    async def _load_document() -> list:
        raw_document = PyPDFium2Loader(
            "app/documents/Nifty Bridge Terms of Service.pdf"
        ).load()
        return raw_document

    @staticmethod
    async def _split_document(raw_document: list) -> list:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        chunked_documents = text_splitter.split_documents(raw_document)
        return chunked_documents

    @staticmethod
    async def _store_documents(chunked_documents: list) -> Chroma:
        vector_store = Chroma.from_documents(chunked_documents, OpenAIEmbeddings())
        return vector_store
