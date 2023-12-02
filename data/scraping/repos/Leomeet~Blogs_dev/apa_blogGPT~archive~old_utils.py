"""
Streamlit components utility file
"""

import streamlit as st
import pathlib
import os
import glob
import requests
import pickle
import functools
import openai
import threading


from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from streamlit_chat import message
from streamlit_toggle import st_toggle_switch
from streamlit.runtime.scriptrunner import add_script_run_ctx
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from requests_auth_aws_sigv4 import AWSSigV4

from prompts import FINAL_PROMPT
from models_config import MODELS_JSON
from vecstore import Vecstore
from exceptions import (
    LlmModelSelectionException,
    EmptyModelSelectionException,
    MissingKeysException,
)


class UiElements(object):
    """
    User interface components and supporting functions
    """

    def __init__(self):
        self.counter = 0

    def select_llm_model(self):
        """
        Function to add llm selection panel in sidebar
        """
        # LLM model selection section
        model_name = st.sidebar.selectbox(
            "Models",
            options=self.MODELS.keys(),
        )
        if model_name:
            st.session_state["selected_model"] = model_name

        # publishing information about the chosen LLM model
        expander = st.sidebar.expander("model info")
        info = self.MODELS.get(st.session_state["selected_model"], "")
        expander.write(info.get("info", ""))
        if info:
            if info["name"] not in ["GPT-3", "GPT-4"] and info.get("conn") == "":
                st.sidebar.warning("Please Enable EndPoint")

    def select_documents(self):
        """
        select documents wrapper
        """
        # Document Selection
        st.sidebar.divider()

        vecstore = Vecstore()
        data_collection = vecstore.list_all_collections()
        data_collection.append("none")
        st.sidebar.subheader("Data Collection")
        selected_dataset = st.sidebar.selectbox(
            "select a dataset",
            options=data_collection,
            on_change=vecstore.release_all(),
        )
        if selected_dataset:
            st.session_state["dataset"] = selected_dataset
            # vecstore.load_collection(selected_dataset)

        document_upload_toggle = st_toggle_switch(
            active_color="grey",
            label="upload document",
        )
        if document_upload_toggle:
            # Document Upload
            self.upload_files()

    def upload_files(self):
        """
        performing file upload for data collection

        Returns:
            file_name: name of the file being uploaded
        """
        uploaded_files = st.sidebar.file_uploader(
            "dataset file",
            accept_multiple_files=True,
        )
        for uploaded_file in uploaded_files:
            data = uploaded_file.getvalue()
            file_name = (uploaded_file.name).split(".")[0]
            self.background_db_upload(file_name, data)
            st.write("filename:", file_name)

    def upload_callback(self, future):
        """
        a callback function for file upload thread to notify when the task is complete

        Args:
            future: Threadpoolexecutor instance
        """
        st.sidebar.success("Uploaded your data")

    def background_db_upload(self, file_name, data):
        """
        data upload to database with vector embeddings using Threading

        Args:
            file_name: name of the file
            data: content of the file
        """
        vecstore = Vecstore()
        if file_name not in vecstore.list_all_collections():
            with ThreadPoolExecutor() as executor:
                future = executor.submit(vecstore.setup_new_collection, file_name, data)
                future.add_done_callback(self.upload_callback)
                for task in executor._threads:
                    add_script_run_ctx(task)

    def clear_conversation(self):
        """
        flushing the session memory with clear button press
        """
        # clear conversation section
        clear_button = st.sidebar.button("Clear Conversation", key="clear")
        if clear_button:
            st.session_state["dataset"] = ""
            st.session_state["generated"] = []
            st.session_state["past"] = []
            st.session_state["messages"] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            st.session_state["number_tokens"] = []
            st.session_state["cost"] = []
            st.session_state["total_cost"] = 0.0
            st.session_state["tokens"] = []

    def submit_query(self):
        """function to set input box empty after hitting enter"""
        st.session_state.query = st.session_state.widget
        st.session_state.widget = ""


class UiWrappers(UiElements):
    """
    contains individual sections of application
    sidebar and chat
    """

    def __init__(self):
        self.MODELS = MODELS_JSON["models"]

    def sidebar(self):
        """
        Base sidebar Container
        """
        st.sidebar.image("images/simform-logo.png", width=60)
        st.sidebar.title("BlogsGPT ✨ (vecstore)")
        self.select_llm_model()
        self.clear_conversation()
        self.select_documents()

    def chat(self):
        """
        Base chat Container
        """
        # container for chat history
        response_container = st.container()
        # container for text box
        container = st.container()
        with container:
            st.text_input("You:", key="widget", on_change=self.submit_query)
            functions = General()
            if st.session_state.query:
                output = (
                    functions.generate_conversational_response(st.session_state.query)
                    if st.session_state["selected_model"].lower() in ["gpt-3", "gpt-4"]
                    else functions.generate_from_custom_api(st.session_state.query)
                )
                st.session_state["past"].append(st.session_state.query)
                st.session_state["generated"].append(output)

        if st.session_state["generated"]:
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    message(
                        st.session_state["past"][i], is_user=True, key=str(i) + "_user"
                    )
                    message(st.session_state["generated"][i], key=str(i))


class General:
    """
    General Utility functions for application
    """

    def __init__(self):
        self.MODELS = MODELS_JSON["models"]
        self.open_ai_key = os.environ.get("OPENAI_API_KEYS", None)
        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
        self.aws_secret_secret_kes = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

    def __call__(self):
        """
        setting on selection values
        """
        if st.session_state["selected_model"] == "":
            raise EmptyModelSelectionException("No Model Selected")
        else:
            models_data = self.MODELS.get(st.session_state["selected_model"], None)
            for i, key in enumerate(models_data.get("keys")):
                if not os.environ.get(key):
                    raise MissingKeysException(f"Missing required keys: {key} ")

    def initialize_session(self):
        """
        initializing session variables
        """
        # Initialise session state variables
        if "dataset" not in st.session_state:
            st.session_state["dataset"] = []
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        if "cost" not in st.session_state:
            st.session_state["cost"] = [
                0.0,
            ]
        if "tokens" not in st.session_state:
            st.session_state["tokens"] = [
                0,
            ]
        if "chat_summary" not in st.session_state:
            st.session_state["chat_summary"] = []
        if "selected_model" not in st.session_state:
            st.session_state["selected_model"] = ""
        if "query" not in st.session_state:
            st.session_state.query = ""

    def generate_from_custom_api(self, query):
        """call custom api mapped with custom llm endpoint

        Args:
            query: user input

        Returns:
            : answer response from custom llm
        """
        info = [
            x for x in self.MODELS if x["name"] == st.session_state["selected_model"]
        ]
        pre_set_url = info[0].get("conn", None) if info else ""

        st.session_state["messages"].append({"role": "user", "content": query})
        payload = {
            "inputs": query,
        }

        aws_auth = AWSSigV4(
            "sagemaker",
            region="us-east-1",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        )

        try:
            ans = requests.request(
                "POST", pre_set_url, auth=aws_auth, json=payload, timeout=5
            )
            if str(ans.status_code)[0] == "4":
                st.warning("Unable to process Request check endpoint")
        except ConnectionError as error:
            print(error)

        ans = ans.json()[0].get("generated_text")
        st.session_state["messages"].append({"role": "ai", "content": ans})

        return (ans,)

    def generate_conversational_response(self, query):
        """
        Generates Answer for given query by calling OpenAI API
        """
        utils = LangchainUtils()
        store = utils.conversational_summary()
        st.session_state["messages"].append({"role": "user", "content": query})
        sources = ""
        if st.session_state["dataset"] != "none":
            with open("custom_embeddings/apa_data_with_source.pkl", "rb") as file:
                index = pickle.load(file)
            sources = utils.doc_search_vecstore(st.session_state["dataset"], query)

        chat_history = st.session_state.get("chat_summary")
        chat_summary = ""
        if chat_history:
            chat_summary = " ".join(x.get("history") for x in chat_history)
        with get_openai_callback() as openai_callback:
            answer = utils.get_answer(sources, query, chat_summary, True)
            st.session_state["tokens"].append(openai_callback.total_tokens)
            st.session_state["cost"].append(openai_callback.total_cost)

            st.session_state["messages"].append(
                {"role": "ai", "content": answer.get("output_text", None)}
            )

        store.save_context(
            inputs={"input": query},
            outputs={"output": answer.get("output_text", None)},
        )
        st.session_state.get("chat_summary").append(store.load_memory_variables({}))

        return answer.get("output_text")

    def generate_static_response(self, query):
        """
        Generating Response based on the query given
        with a similarity search to given doc / dataset

        Args:
            query (str): Question by user

        Returns:
            str: answer from LLM
        """
        utils = LangchainUtils()
        st.session_state["messages"].append({"role": "user", "content": query})
        with open("custom_embaddings/apa_data_with_source.pkl", "rb") as f:
            index = pickle.load(f)
        sources = utils.search_docs(index, query)
        with get_openai_callback() as openai_callback:
            answer = utils.get_answer(sources, query, True)
            st.session_state["tokens"].append(openai_callback.total_tokens)
            st.session_state["cost"].append(openai_callback.total_cost)
            st.session_state["messages"].append(
                {"role": "ai", "content": answer.get("output_text", None)}
            )

        return answer.get("output_text")

    def get_chat_current_info():
        cost = st.session_state["cost"]
        tokens = st.session_state["tokens"]
        return cost[-1], tokens[-1]

    def get_chat_total_info():
        cost = functools.reduce(lambda a, b: a + b, st.session_state["cost"])
        tokens = functools.reduce(lambda a, b: a + b, st.session_state["tokens"])
        return cost, tokens


class LangchainUtils(General):
    """
    Langchain and embeddings utility functions
    """

    def doc_search_faiss(self, index, query):
        """searching for similar embeddings in provided index
        : index - pkl or faiss embedding file data
        : query - query embeddings to compare to

        Returns:
            : document sources
        """
        response = openai.Embedding.create(model="text-embedding-ada-002", input=query)
        embedding = response["data"][0]["embedding"]
        docsearch = index.similarity_search_by_vector(embedding, k=2)
        return docsearch

    def doc_search_vecstore(self, index: str, query: str):
        """vector database search with index / collection name and given query

        Args:
            index (str): name of the collection
            query (str): asked question

        Returns:
            obj: collection of relevant documents
        """
        vecstore = Vecstore()
        vecstore.load_collection(index)
        docsearch = vecstore.search_with_index(index, query)
        return self.format_docsearch(docsearch)

    def format_docsearch(self, docsearch):
        """formatting the milvus result object to a list of document object for langchain

        docsearch: milvus result object
        """
        result_object = []
        for result in docsearch[0]:
            # here content is the name of my output field that holds text information
            text_info = result.entity.get("content")
            doc = Document(page_content=text_info, metadata={"source": result.score})
            # pair = (doc, result.score)
            result_object.append(doc)
        ()
        return result_object

    def get_answer(self, docsearch, query, chat_history, is_user_uploaded_data):
        """generates ans with api call to OpenAI

        Args:
            docsearch: searched documents sources
            query: question asked by user
            chat_history : summary of previous messages
            is_user_uploaded_data (bool): to search with prompt template

        Returns:
            response from api call
        """
        llm = OpenAI(temperature=0.7, openai_api_key=self.open_ai_key)

        if is_user_uploaded_data:
            search_chain = load_qa_with_sources_chain(
                llm=llm,
                chain_type="stuff",
                prompt=FINAL_PROMPT,
            )
        else:
            search_chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")

        answer = search_chain(
            {
                "input_documents": docsearch,
                "question": query,
                "history": chat_history,
            },
            return_only_outputs=True,
        )

        return answer

    def get_sources(self, answer, docs, is_user_uploaded_data):
        """gets sources information from given answer

        Args:
            answer (_type_): answer generated by ai
            docs (_type_): document to search into
            is_user_uploaded_data (bool): custom data

        Returns:
            : returns sources of generated ans
        """
        source_keys = list(answer["output_text"].split("SOURCES: ")[-1].split(","))

        if not is_user_uploaded_data:
            return source_keys

        source_docs = []
        for doc in docs:
            if doc.metadata["source"] in source_keys:
                source_docs.append(doc)

        return source_docs

    def wrap_text_in_html(self, text) -> str:
        """Wraps each text block separated by newlines in <p> tags"""
        if isinstance(text, list):
            text = "\n<hr/>\n".join(text)

        return "".join([f"<p>{line}</p>" for line in text.split("\n")])
