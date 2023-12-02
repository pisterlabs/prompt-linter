import os
from enum import Enum
from typing import List, Optional, Tuple
import openai
from actionweaver import ActionHandlerMixin, RequireNext, SelectOne, action
from actionweaver.llms.openai.chat import OpenAIChatCompletion
from actionweaver.llms.openai.tokens import TokenUsageTracker
from llama_index import (
    Document,
    ServiceContext,
    SimpleDirectoryReader,
    SimpleWebPageReader,
    VectorStoreIndex,
)

openai.api_key = os.getenv("OPENAI_API_KEY")


class RAGBot(ActionHandlerMixin):
    def __init__(self, logger, st):
        self.index = VectorStoreIndex.from_documents([])
        self.logger = logger
        self.st = st
        self.token_tracker = TokenUsageTracker(budget=3000, logger=logger)
        self.llm = OpenAIChatCompletion(
            "gpt-3.5-turbo-16k-0613",
            token_usage_tracker=self.token_tracker,
            logger=logger,
        )

        self.init_messages()

    def init_messages(self):
        system_str = "You are a helpful assistant. Please do not try to answer the question directly."
        self.messages = [{"role": "system", "content": system_str}]

    def contains_url(self, text):
        import re

        # Regular expression pattern for matching URLs
        url_pattern = r"https?://\S+|www\.\S+"

        # Search for URLs in the input text
        if re.search(url_pattern, text):
            return True
        else:
            return False

    def __call__(self, query):
        self.messages.append(
            {"role": "user", "content": query},
        )

        return self.llm.create(
            self.messages,
            stream=True,
        )

    @action(name="AnswerQuestion", stop=True)
    def answer_question(self, query: str):
        """
        Answer a question or search online.

        Parameters
        ----------
        query : str
            The query to be used for answering a question.
        """

        context_str = self.recall(query)
        context = (
            "Information from knowledge base:\n"
            "---\n"
            f"{context_str}\n"
            "---\n"
            f"User: {query}\n"
            "Only answer question based on information from knowledge base"
            "If you don't have information in the knowledge base, performs a Google search instead. Your Response:"
        )

        return self.llm.create(
            [
                {"role": "user", "content": context},
            ],
            orch_expr=SelectOne(["GoogleSearch"]),
        )

    @action(name="ExecuteInstruction", stop=True)
    def execute_instruction(self, query: str):
        """
        Execute user instruction and provide an appropriate response.

        e.g. translate the above text to French

        Parameters
        ----------
        query : str
            The user's request or instruction to be executed.
        """

        return self.llm.create(
            self.messages,
            stream=True,
            orch_expr=SelectOne([]),
        )

    @action(name="GoogleSearch", stop=True, scope="search")
    def search(self, query: str):
        """
        Perform a Google search and return query results with titles and links.

        Parameters
        ----------
        query : str
            The search query to be used for the Google search.

        Returns
        -------
        str
            A formatted string containing Google search results with titles, snippets, and links.
        """

        with self.st.spinner(f"Searching '{query}'..."):
            from langchain.utilities import GoogleSearchAPIWrapper

            search = GoogleSearchAPIWrapper()
            res = search.results(query, 10)
            formatted_data = ""

            # Iterate through the data and append each item to the formatted_data string
            for idx, item in enumerate(res):
                formatted_data += f"({idx}) {item['title']}: {item['snippet']}\n"
                formatted_data += f"[Source]: {item['link']}\n\n"

        return f"Here are Google search results:\n\n{formatted_data}"

    @action("Recall", stop=True)
    def recall(self, text):
        """
        Recall info from your knowledge base.

        Parameters
        ----------
        text : str
            The query text used to search the agent's knowledge base.

        Returns
        -------
        str
            A response containing relevant information retrieved from the knowledge base along with sources.
            If no information is found, it returns "No information on that topic."
        """

        query_engine = self.index.as_query_engine()
        response = query_engine.query(text)

        sources = []
        for v in response.metadata.values():
            sources += v["source"]

        sources = list(set(sources)) if type(sources) == list else sources

        if response.response:
            return f"{response.response}"
        else:
            return "No information on this topic."

    @action("Read", stop=True)
    def read(self, sources: str):
        """
        Read content from various sources.

        Parameters
        ----------
        sources : str
            The source identifier, which can be a web link or a file path, e.g. "https://www.example.com", "/path/to/your/local/file.txt".
        """
        if self.contains_url(sources):
            return self.llm.create(
                [{"role": "user", "content": sources}],
                orch_expr=RequireNext(["ReadURL"]),
            )

        return self.llm.create(
            [{"role": "user", "content": sources}],
            orch_expr=RequireNext(["ReadFile"]),
        )

    @action("ReadURL", scope="read", stop=True)
    def read_url(self, urls: List[str]):
        """
        Read the content from the provided web links.

        Parameters
        ----------
        urls : List[str]
            List of URLs to scrape.

        Returns
        -------
        str
            A message indicating successful reading of content from the provided URLs.
        """

        with self.st.spinner(f"Learning the content in {urls}"):
            service_context = ServiceContext.from_defaults(chunk_size=512)
            documents = SimpleWebPageReader(html_to_text=True).load_data(urls)

            for doc in documents:
                doc.metadata = {"source": urls}
                self.index.insert(
                    doc,
                    service_context=service_context,
                )
        return f"Contents in URLs {urls} have been successfully learned."

    @action("ReadFile", scope="read", stop=True)
    def read_file(
        self,
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        *args,
        **kwargs,
    ):
        """
        Read the content from provided files or directories.

        Parameters
        ----------
        input_dir : str, optional
            Path to the directory (default is None).
        input_files : List, optional
            List of file paths to read (overrides input_dir if provided).

        Returns
        -------
        str
            A message indicating successful reading of content from the files.
        """
        reader = SimpleDirectoryReader(input_dir=input_dir, input_files=input_files)
        with self.st.spinner(
            f"Learning the content in {[str(file) for file in reader.input_files]}"
        ):
            service_context = ServiceContext.from_defaults(chunk_size=512)
            documents = reader.load_data()
            for doc in documents:
                doc.metadata = {"source": input_dir or str(input_files)}
                self.index.insert(doc, service_context=service_context)
        return f"Contents in files {[str(file) for file in reader.input_files]} have been successfully embedded."

    @action("Remember", stop=True)
    def remember_convos_and_clear_messages(self, text_list: List[str]) -> str:
        """
        Remember contents from a list of text documents.

        Parameters
        ----------
        text_list : List[str]
            List of text documents to be stored.

        Returns
        -------
        str
            A message indicating that the contents have been successfully stored.
        """

        with self.st.spinner("Remembering ... "):
            service_context = ServiceContext.from_defaults(chunk_size=512)
            documents = [Document(text=t) for t in text_list]

            for doc in documents:
                doc.metadata = {"source": "previous conversation"}
                self.index.insert(doc, service_context=service_context)

            self.init_messages()

        return "Contents have been successfully embedded."


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="bot.log",
        filemode="a",
        format="%(asctime)s.%(msecs)04d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    agent = RAGBot(logger, None)
