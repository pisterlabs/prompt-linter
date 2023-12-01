from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from constants import Constants
import streamlit as st
import os

class DataReader:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.storage_dir = Constants.DATA_STORAGE_DIR
        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def load_index_if_exists(self, service_context):
        index_file = os.path.join(self.storage_dir, 'docstore.json')
        if os.path.isfile(index_file):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                return load_index_from_storage(storage_context, service_context=service_context)
            except Exception as e:
                print(f"Error loading index from {index_file}: {e}")
        return None

    def load_data(self, llm_model, llm_temp):
        with st.spinner(Constants.DATA_LOADING_TEXT):
            service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model, max_tokens=Constants.MAX_TOKENS, temperature=llm_temp, system_prompt=Constants.SYSTEM_PROMPT))

            index = self.load_index_if_exists(service_context)
            if index:
                return index

            reader = SimpleDirectoryReader(input_dir=self.input_dir, recursive=True)
            docs = reader.load_data()
            index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)

            # storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index.storage_context.persist(persist_dir=self.storage_dir)
            return index
