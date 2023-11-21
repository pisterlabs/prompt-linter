# -*- coding: utf-8 -*-
import asyncio
import json
import re

from web_gpt.open_ai_helper import OpenAiHelper
from web_gpt.langchain_helper import LangChainHelper


class WebGPT(OpenAiHelper, LangChainHelper):
    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            vector_store_model: str = "gpt-3.5-turbo-16k",
            prompt: str = """
            Use the following context snippets to answer the question at the end.
            Instructions:
            - Carefully read the document and analyze it fully understanding its meaning.
            - Conduct a hermeneutic analysis, exploring in depth the textual and contextual layers of the document.
            - Answer the question in as much detail as possible to the user's question, as much as the context allows.
            - If you don't know the answer, just say you don't know, don't try to come up with an answer.
            - Give fragments of the context from where you got the information.
            - Always answer in the form of a marking list.
            - Study the document, get into its essence, reveal hidden meanings and subtext.
            {context}
            """,
            urls_count: int = 1,
            search_region: str = "ru-ru"
    ):
        """
        Class based on Open air API and Langchain, seamlessly connecting ChatGPT to the Internet
        :param model: chat model
        :param vector_store_model: the model that will search the site, "gpt-3.5-turbo-16k" is recommended
        :param prompt: instructions for the neural network on how to work with the document, you must add {context} to the end, it is recommended to leave the default
        :param urls_count: the number of links from which the neural network will take information, it is recommended no more than 3
        :param search_region: region for internet search, for example "ru-ru"
        """
        self.urls_count = urls_count
        self.messages = None
        self.open_ai_key = self.get_open_ai_key()
        self.loop = asyncio.get_event_loop()
        self.prompt = prompt
        OpenAiHelper.__init__(self, model)
        LangChainHelper.__init__(self, vector_store_model, search_region)
        self.open_ai_key = self.get_open_ai_key()

    async def ask(self, messages: list) -> dict:
        """
        A method of the Class based on a request to Web GT, if the user's response requires information from the Internet, it will launch LangChain methods, otherwise
        WebGPT will give a normal response

        The response is returned as a dict:
        If the response did not require the Internet: {"type": "def", "content": "Response"}
        If the response required the Internet: {"type": "web", "content": "Response", "vectorstore": "formatted information source"}

        :param messages: list of messages, more details https://platform.openai.com/docs/api-reference/chat
        """
        self.messages = messages
        chat_completion = await self.chat_completion(messages=self.messages)
        try:
            if chat_completion["choices"][0]["message"]["content"] is not None:
                return {"type": "def", "content": chat_completion["choices"][0]["message"]["content"]}
            llm_answer = await self.vector_store_asq(json.loads(chat_completion["choices"][0]["message"]["function_call"]["arguments"])["query"])
            return {"type": "web", "content": llm_answer["content"], "vectorstore": llm_answer["vectorstore"]}
        except Exception as e:
            raise Exception(e)

    async def vector_store_asq(self, query=None, old_vectorstore=None, messages: list = None) -> dict:
        """
        A method of the Class based on a request to an already existing vectorstore
        WebGPT will give a normal response

        The response is returned as a dict:
        {"content": "Response", "vectorstore": vectorstore}

        :param old_vectorstore: formatted information source that is returned in the ask method
        :param messages: list of messages, more details https://platform.openai.com/docs/api-reference/chat
        """
        template = self.prompt
        if old_vectorstore is None:
            query = await self.find_links(query)
            if len(query["text"]) < 2:
                query["text"] = "Очень подробно суммаризируй эту статью"
            vector_data = await self.create_index(query)
            vectorstore = vector_data[0]
            old_vectorstore = vector_data[1]
            last_query = query["text"]
            template += "\n\nВопрос пользователя: {question}\nОтвет полезного помощника, который следует инструкциям:"
        else:
            vectorstore = (await self.create_index(all_splits=old_vectorstore))[0]
            self.messages = messages
            if len(self.messages) > 1:
                for message in self.messages[:-1]:
                    if message["role"] == "user":
                        template += f"Вопрос пользователя: {message['content']}"
                    else:
                        template += f"Ответ полезного помощника, который следует инструкциям: {message['content']}"
            template += "Вопрос пользователя: {question}\nОтвет полезного помощника, который следует инструкциям:"
            last_query = self.messages[-1]["content"]

        result = await self.llm_asq(template=template, vectorstore=vectorstore, last_query=last_query)

        return {"content": result["result"], "vectorstore": old_vectorstore}

    @staticmethod
    async def find_links(text):
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        matches = re.findall(pattern, text)
        if len(matches) != 0:
            return {"text": text.replace(matches[0], ""), "url": matches[0]}
        return {"text": text, "url": None}
