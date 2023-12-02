
# -*- coding: utf-8 -*-
"""
****************************************************
*                    LLM Tutor                     *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import copy
from typing import Any, List, Tuple
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from uuid import uuid4
from src.configuration import configuration as cfg
from src.control.chroma_knowledgebase_controller import ChromaKnowledgeBase, EmbeddingFunction, Embeddings, Document
from src.utility.bronze import json_utility
from src.utility.silver import file_system_utility


class TutorController(object):
    """
    Class for controlling the main process.
    """

    def __init__(self, config: dict = None) -> None:
        """
        Initiation method.
        :paran config: Configuration for controller instanciation.
            Defaults to None in which case a new controller is instanciated.
        """
        self.config = {
            "conversations": {}
        } if config is None else config
        self.llm = None if config is None else self.load_general_llm(
            **config["llm"])
        self.llm_type = None if config is None else config["llm"]["model_type"]
        self.kb = None if config is None else self.load_knowledge_base(
            **config["kb"])
        self.doc_types = {
            "base": {"splitting": None}
        } if config is None else config["doc_types"]
        self.conversations = {}
        if config is not None:
            for conversation in config["conversations"]:
                self.start_conversation(
                    use_uuid=conversation, document_type=config["conversations"][conversation]["document_type"])
        self.temporary_config = copy.deepcopy(self.config)

    def save_config(self, config_name: str) -> None:
        """
        Method for saving config.
        :param config_name: Name to save config under.
        """
        self.config = copy.deepcopy(self.temporary_config)
        path = os.path.join(cfg.PATHS.CONFIG_PATH, f"{config_name}.json")
        if os.path.exists(path):
            os.remove(path)
        json_utility.save(self.config, path)

    def get_available_configs(self) -> List[str]:
        """
        Method for retrieving a list of available configs.
        :return: List of config names.
        """
        return [file.replace(".json", "").replace(cfg.PATHS.CONFIG_PATH, "")[1:] for file in file_system_utility.get_all_files(cfg.PATHS.CONFIG_PATH) if file.endswith(".json")]

    def load_config(self, config_name: str) -> None:
        """
        Method for loading config.
        :param config_name: Name of config to load.
        """
        path = os.path.join(cfg.PATHS.CONFIG_PATH, f"{config_name}.json")
        config = json_utility.load(path)
        self.__init__(config)

    def load_general_llm(self, model_path: str, model_type: str = "llamacpp") -> None:
        """
        Method for (re)loading main LLM.
        :param model_path: Model path.
        :param model_type: Model type frmo 'llamacpp', 'chat', 'instruct'. Defaults to 'llamacpp'.
        """
        self.temporary_config["llm"] = {
            "model_path": model_path, "model_type": model_type}
        if model_type == "llamacpp":
            self.llm = LlamaCpp(
                model_path=model_path,
                verbose=True,
                n_ctx=2048)

    def load_knowledge_base(self, kb_path: str, kb_base_embedding_function: EmbeddingFunction = None) -> None:
        """
        Method for loading knowledgebase.
        :param kb_path: Folder path to knowledgebase.
        :param kb_base_embedding_function: Base embedding function to use for knowledgebase. 
            Defaults to None in which case the knowledgebase default is used.
        """
        # TODO: Utilize configuration to instanciate embedding functions.
        self.temporary_config["kb"] = {
            "kb_path": kb_path}
        self.kb = ChromaKnowledgeBase(
            peristant_directory=kb_path, base_embedding_function=kb_base_embedding_function)

    def register_document_type(self, document_type: str, embedding_function: EmbeddingFunction = None, splitting: Tuple[int] = None) -> None:
        """
        Method for registering a document type.
        :param document_type: Name to identify the documen type.
        :param embedding_function: Embedding function the document type. Defaults to base embedding function.
        :param splitting: A tuple of chunk size and overlap for splitting. Defaults to None in which case the documents are not split.
        """
        self.doc_types[document_type] = {"splitting": splitting}
        self.kb.get_or_create_collection(
            document_type, embedding_function)

    def load_documents(self, documents: List[Document], document_type: str = None) -> None:
        """
        Method for loading documents into knowledgebase.
        :param documents: Documents to load.
        :param document_type: Name to identify the documen type. Defaults to "base".
        """
        self.kb.embed_documents(
            name="base" if document_type is None else document_type, documents=documents)

    def load_files(self, file_paths: List[str], document_type: str = None) -> None:
        """
        Method for (re)loading folder contents.
        :param folder: Folder path.
        :param document_type: Name to identify the documen type. Defaults to "base".
        """
        document_type = "base" if document_type is None else document_type
        self.kb.load_files(file_paths, document_type, self.doc_types.get(
            document_type, {}).get("splitting"))

    def start_conversation(self, use_uuid: str = None, document_type: str = None) -> str:
        """
        Method for starting conversation.
        :param use_uuid: UUID to start conversation under. Defaults to newly generated UUID.
        :param document_type: Target document type. Defaults to None in which case "base" is set.
        :return: Conversation UUID.
        """
        use_uuid = str(uuid4()) if use_uuid is None else use_uuid
        document_type = "base" if document_type is None else document_type
        self.temporary_config["conversations"][use_uuid] = {
            "document_type": document_type
        }

        memory = ConversationBufferMemory(
            memory_key=f"chat_history", return_messages=True)
        self.conversations[use_uuid] = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.kb.get_retriever(
                "base" if document_type is None else document_type),
            memory=memory
        )
        return use_uuid

    def conversational_query(self, conversation_uuid: str, query: str) -> dict:
        """
        Method for querying via conversation.
        :param conversation_uuid: UUID of conversation to run query on.
        :param query: Query to run.
        :return: Query results.
        """
        if conversation_uuid not in self.conversations:
            conversation_uuid = self.start_conversation(
                use_uuid=conversation_uuid)
        result = self.conversations[conversation_uuid]({"question": query})
        result["conversation"] = conversation_uuid
        self.temporary_config["conversations"][conversation_uuid]["result"] = result
        return result

    def query(self, query: str, document_type: str = None, include_source: bool = True) -> dict:
        """
        Method for direct querying.
        :param query: Query to run.
        :param document_type: Target document type. Defaults to None in which case "base" is set.
        :param include_source: Flag for declaring whether to include source. Defaults to True.
        :return: Query results.
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.kb.get_retriever(
                "base" if document_type is None else document_type),
            return_source_documents=include_source)(query)
