import os
import sys
import pyttsx3
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from res import API_Hidden_Key as Key
from AppOpener import open as open_app, close as close_app
from youtubesearchpython import VideosSearch
import json
import re

os.environ["OPENAI_API_KEY"] = Key.APIKEY


class ZapyBot:
    def __init__(self):
        self.is_persist_enabled = False
        self.query = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        }

        if len(sys.argv) > 1:
            self.query = sys.argv[1]

        self.index = self.persist_index(self.is_persist_enabled)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=self.index.vectorstore.as_retriever(search_kwargs={"K": 1}),
        )
        with open("res/data/commands.json") as f:
            self.command_patterns = json.load(f)
        self.chat_history = []
        self.version = "v3.0"
        self.name = "Zapy"

    def ask(self, question):
        que = question
        result = self.chain({"question": que, "chat_history": self.chat_history})
        self.chat_history.append((que, result["answer"]))
        return result["answer"]

    @staticmethod
    def persist_index(persist_flag: bool):
        if persist_flag and os.path.exists("persist"):
            print("Reusing index...\n")
            vectorstore = Chroma(
                persist_directory="persist", embedding_function=OpenAIEmbeddings()
            )
            ind = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            loader = DirectoryLoader(
                "zapy_data/data/", show_progress=True, recursive=True, glob="*.txt"
            )
            if persist_flag:
                ind = VectorstoreIndexCreator(
                    vectorstore_kwargs={"persist_directory": "persist"}
                ).from_loaders([loader])
            else:
                ind = VectorstoreIndexCreator().from_loaders([loader])
        return ind

    @staticmethod
    def say(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def commands(self, question):
        for command, details in self.command_patterns.items():
            pattern = details["pattern"]
            action = details["action"]
            match = re.match(pattern, question.lower())
            if match:
                return getattr(self, action)(*match.groups())
        return self.ask(question)

    @staticmethod
    def open_app(app_name):
        open_app(app_name, match_closest=True, output=False)
        return "Opening " + app_name

    @staticmethod
    def close_app(app_name):
        close_app(app_name, match_closest=True, output=False)
        return "Closing " + app_name

    def search_youtube(self, question, max_results=5):
        query = question.split("youtube ")[-1]
        videos_search = VideosSearch(query, limit=max_results)
        results = videos_search.result()

        if "result" in results:
            videos = results["result"]
            search_results = []
            for video in videos:
                title = video["title"]
                link = video["link"]
                search_results.append({"title": title, "link": link})

            return search_results
