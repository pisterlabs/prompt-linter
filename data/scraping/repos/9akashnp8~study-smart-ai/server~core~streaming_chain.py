from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationChain, ConversationalRetrievalChain

import asyncio
from decouple import config
from functools import cache
from sse_starlette import ServerSentEvent

from utils.constants import DB_DIR
from core.functions import get_chat_history
from core.prompts import CHAT_PROMPT

class StreamingConversationChain:
    memory = ConversationBufferMemory(memory_key="chat_history")
    embedding = OpenAIEmbeddings(openai_api_key=config('OPENAI_API_KEY'))

    def __init__(self, openai_api_key: str, temparature: float) -> None:
        self.openai_api_key = openai_api_key
        self.temperature = temparature

    @cache
    def get_vectorstore(self, collection_name: str) -> Chroma:
        # TODO: Handle collection doesn't exist
        return Chroma(
            collection_name=collection_name,
            persist_directory=DB_DIR,
            embedding_function=self.embedding
        )

    async def generate_response(self, message: str, collection_name: str = None):
        vectorstore = self.get_vectorstore(collection_name)
        callback_handler = AsyncIteratorCallbackHandler()
        llm = OpenAI(
            callbacks=[callback_handler],
            streaming=True,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            get_chat_history=get_chat_history
        )
        conversation = ConversationChain(
            llm=llm,
            memory=self.memory,
            prompt=CHAT_PROMPT,
        )

        if message.startswith("/doc"):
            run = asyncio.create_task(qa.arun({"question": message}))
        run = asyncio.create_task(conversation.apredict(input=message))

        async for token in callback_handler.aiter():
            yield ServerSentEvent(data=token, event='message')
        
        if callback_handler.done.is_set():
            yield ServerSentEvent(data='', event='end')

        await run