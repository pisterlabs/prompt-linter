import asyncio

from duckduckgo_search import DDGS

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class LangChainHelper:
    def __init__(self, vector_store_model: str, search_region: str):
        self.vector_store_model = vector_store_model
        self.llm = ChatOpenAI(model_name=self.vector_store_model, temperature=0, max_tokens=10000)
        self.messages = None
        self.loop = asyncio.get_event_loop()
        self.search_region = search_region

    async def llm_asq(self, template: str, vectorstore, last_query: str):
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": PromptTemplate.from_template(template)})
        return await self.loop.run_in_executor(None, qa_chain, {"query": last_query + "\n\nОтвечай на русском"})

    async def create_index(self, query=None, all_splits=None):
        if not all_splits:
            if query["url"] is None:
                url = await self.get_urls(query["text"])
            else:
                url = query["url"]
            data = Html2TextTransformer().transform_documents(
                await self.loop.run_in_executor(None, AsyncChromiumLoader([url]).load))
            all_splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(data)
        vectorstore = await self.loop.run_in_executor(None, Chroma.from_documents, all_splits, OpenAIEmbeddings())
        return vectorstore, all_splits

    async def get_urls(self, query):
        with DDGS() as search:
            for r in search.text(query, region=self.search_region, safesearch='off', timelimit='y'):
                return r['href']
