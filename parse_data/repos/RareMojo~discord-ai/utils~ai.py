from typing import TYPE_CHECKING

import pinecone
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate)
from langchain.vectorstores import Pinecone

from discord_bot.logger import log_debug, log_error, log_info

if TYPE_CHECKING:
    from discord_bot.bot import Bot


class ChatAgent:
    """
    A class for managing a conversation with a bot.
    """

    def __init__(self, bot: "Bot", channel_id: str, temperature: float=0, return_messages: bool=True):
        """
        Initializes a ChatAgent instance.
        Args:
          bot (Bot): The bot instance.
          channel_id (str): The channel ID.
          temperature (float): The temperature for OpenAI predictions.
          return_messages (bool): Whether to return messages.
        Side Effects:
          Reads the preprompt from the configs directory.
        Notes:
          The preprompt is replaced with the bot's persona.
        """
        self.bot = bot
        client = self.bot.openai_api_key
        model = self.bot.openai_model
        self.channel_id = channel_id

        with open(self.bot.paths["configs"] / "preprompt", "r") as f:
            preprompt = f.read()

        preprompt = preprompt.replace("{persona}", bot.config.get("persona"))

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(f"{preprompt}"),
                MessagesPlaceholder(variable_name=channel_id),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.llm = ChatOpenAI(client=client, model=str(model), temperature=temperature)
        memory = ConversationBufferWindowMemory(
            k=3, memory_key=channel_id, return_messages=return_messages
        )
        self.conversation = ConversationChain(
            memory=memory, prompt=self.prompt, llm=self.llm, verbose=True
        )

    def predict(self, prompt: str):
        """
        Predicts a response to a prompt.
        Args:
          prompt (str): The prompt to respond to.
        Returns:
          str: The predicted response.
        Examples:
          >>> agent.predict("Hello!")
          "Hi there!"
        """
        response = self.conversation.predict(input=prompt)
        return response


class ChatQuery:
    """
    Class for creating a query for a chatbot.
    """

    def __init__(self, bot: "Bot", namespace: str):
        """
        Initializes the ChatQuery class.
        Args:
          bot (Bot): The bot object.
          namespace (str): The namespace for the query.
        Side Effects:
          Initializes the LLM, QA Prompt, LLM Chain, ChatOpenAI, OpenAIEmbeddings, Pinecone, and ConversationalRetrievalChain objects.
        """
        log_debug(bot, "Loading LLM Query")
        self.llm = OpenAI(temperature=0, openai_api_key=bot.openai_api_key)
        self.streaming_llm = ChatOpenAI(
            streaming=True,
            model_name=bot.openai_model,
            openai_api_key=bot.openai_api_key,
            temperature=0,
            verbose=True,
        )

        QA_V2 = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        Very Important: If the question is about writing code use backticks (```) at the front and end of the code snippet and include the language use after the first ticks.
        If you don't know the answer, just say you don't know. DO NOT allow made up or fake answers.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail when as possible when responding.
        Now, let's think step by step and get this right:

        {context}

        Question: {question}
        All answers should be in MARKDOWN (.md) Format:"""

        self.qap = PromptTemplate(
            template=QA_V2, input_variables=["context", "question"]
        )

        CD_V2 = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        All answers should be in MARKDOWN (.md) Format:
        Standalone question:"""

        self.cdp = PromptTemplate.from_template(CD_V2)

        self.question_generator = LLMChain(llm=self.llm, prompt=self.cdp)
        self.doc_chain = load_qa_chain(
            self.streaming_llm, chain_type="stuff", prompt=self.qap
        )

        pinecone.init(api_key=bot.pinecone_api_key, environment=bot.pinecone_env)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=bot.openai_api_key
        )
        self.vectorstore = Pinecone.from_existing_index(
            index_name=bot.pinecone_index,
            embedding=self.embeddings,
            text_key="text",
            namespace=namespace,
        )

        self.qa = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            combine_docs_chain=self.doc_chain,
            return_source_documents=True,
            question_generator=self.question_generator,
            verbose=True,
        )

    def query(self):
        """
        Returns the ConversationalRetrievalChain object.
        Returns:
          ConversationalRetrievalChain: The ConversationalRetrievalChain object.
        """
        return self.qa
