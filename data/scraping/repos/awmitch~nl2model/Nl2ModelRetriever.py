from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from .Nl2Modelica import ModelObject
from discord import TextChannel, Message, Client
import asyncio
import nest_asyncio
nest_asyncio.apply()
from langchain.schema import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever

class ModelicaRetriever(BaseRetriever):
    modelica_model: ModelObject
    discord_channel: TextChannel = None
    discord_client: Client
    compression_retriever: ContextualCompressionRetriever
    
    def __init__(self, 
                 modelica_model: ModelObject, 
                 discord_client: Client,
                 compression_retriever: ContextualCompressionRetriever,
                 ):
        super().__init__()
        self.modelica_model = modelica_model
        self.discord_client = discord_client
        self.compression_retriever = compression_retriever

    def set_channel(self, channel: TextChannel, user):
        self.discord_channel = channel
        self.user = user

    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """

        docs = asyncio.run(self.ask_question_and_wait_for_answer(query))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """
        question = "Your question here..."
        answer = await self.ask_question_and_wait_for_answer(question)
        return answer

    async def ask_question_and_wait_for_answer(self, question: str) -> str:
        def check(message: Message) -> bool:
            return message.author == self.user

        await self.discord_channel.send(f"{self.user.mention}, {question} Response with answer, \"!ref\", or \"None\"")

        try:
            message = await self.discord_client.wait_for('message', check=check, timeout=60.0)  # Wait for 1 minute
        except asyncio.TimeoutError:
            docs = []
        else:
            command = "!ref"
            none_command="None"
            if message.content.startswith(command):
                docs = self.compression_retriever.get_relevant_documents(question)
            elif message.content.startswith(none_command):
                docs = []
            else:
                docs = [Document(page_content=question,metadata=""),
                        Document(page_content=message.content,metadata=""),]
        return docs
