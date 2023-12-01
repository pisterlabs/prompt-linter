import streamlit as st
from langchain import OpenAI
from llama_index import (GPTVectorStoreIndex, LLMPredictor,
                         QuestionAnswerPrompt, ServiceContext, StorageContext,
                         load_index_from_storage)

STORAGE_DIR = "./storage/"

class QAResponseGenerator:
    def __init__(self, selected_model, pdf_reader):
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=selected_model, openai_api_key=st.secrets["OPENAI_API_KEY"]))
        self.pdf_reader = pdf_reader
        self.QA_PROMPT_TMPL = (
            "下記の情報が与えられています。 \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "この情報を参照して次の質問に答えてください: {query_str}\n"
        )
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

    def generate(self, question, file_name):
        documents = self.pdf_reader.load_data(file_name)
        try:
            storage_context = StorageContext.from_defaults(persist_dir=f"{STORAGE_DIR}{file_name}")
            index = load_index_from_storage(storage_context)
            print("load existing file..")
        except:
            index = GPTVectorStoreIndex.from_documents(documents, service_context=self.service_context)
            index.storage_context.persist(persist_dir=f"{STORAGE_DIR}{file_name}")
        
        engine = index.as_query_engine(text_qa_template=QuestionAnswerPrompt(self.QA_PROMPT_TMPL))
        result = engine.query(question)
        return result.response.replace("\n", ""), result.get_formatted_sources(1000)

