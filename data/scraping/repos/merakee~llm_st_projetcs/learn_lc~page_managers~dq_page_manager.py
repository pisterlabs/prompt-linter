# pyhton
# import os
# import sys
# import re
from enum import Enum

# local utility
from st_code.st_session_manager import STSessionManager, STSessionKeys
from app_managers.auth_manager import DeploymentManager
from app_managers.llm_manager import LLMManager
from app_managers.embedding_manager import EmbeddingType, EmbeddingManager
from app_managers.vectordb_manager import RetrieverSearchType, VectorDBManager
from app_managers.langchain_util import DocumentUtil

# LLM
# from langchain.llms import OpenAI
# from langchain.llms.fake import FakeListLLM
# from langchain.llms import HuggingFaceHub

# document loader
# from langchain.docstore.document import Document
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import DirectoryLoader

# splitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding
# from langchain.embeddings import HuggingFaceHubEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings

# Vector store
# from langchain.vectorstores import Chroma

# Prompt
# from langchain.prompts import PromptTemplate
# from langchain.prompts import FewShotPromptTemplate
# from langchain.prompts.example_selector import LengthBasedExampleSelector

# Langchain
# from langchain import hub
# , LLMMathChain, TransformChain, SequentialChain
# from langchain.chains import LLMChain
from langchain.chains import RetrievalQA


# design steps
# 1. load a text document: size constraint
# 2. Chunk it with text splitter
# 3. Generate embegging
# 4. Store embedding in local store
# 5. Get quesitions
# 6. Retrive relevant vetors
# 7. Generate context
# 8. Create prompt
# 9. Get answer and display
# 10. Improve
#       1. List already uploaded files and ablity to pick
#       2. Show source docs
#       3. Interactive answers

class DQSessionKeys(Enum):
    dq_app_state = 1
    dq_file_list = 2
    dq_vectordb_manager = 3
    dq_query_text = 4


class DQSettings:
    class DQAppState:
        upload = "dq_app_state_upload"
        dbready = "dq_app_state_dbready"

    class Embeddings:
        chunk_size_key = "dq_slider_chunk_size"
        chunk_size_min = 50
        chunk_size_max = 8000
        chunk_size_step = 100
        chunk_size_default = 100

        chunk_overlap_key = "dq_slider_chunk_overlap"
        chunk_overlap_min = 0
        chunk_overlap_max = 50
        chunk_overlap_step = 1
        chunk_overlap_default = 10

        embedding_type_key = "dq_selectbox_embedding_type"
        embedding_types = [v.name for v in EmbeddingType]
        embedding_type_default = EmbeddingType.HuggingFaceLocal

    class Retriever:
        search_type_key = "dq_selectbox_search_type"
        search_type_default = RetrieverSearchType.similarity.name
        search_types = [v.name for v in RetrieverSearchType]
        max_match_key = "dq_slider_max_match"
        max_match_min = 1
        max_match_max = 10
        max_match_step = 1
        max_match_default = 3

    class LLMResponse:
        cite_source_key = "dq_checkbox_cite_source"
        cite_source_default = False

    class Info:
        embedding_total_key = "dq_embedding_total"
        embedding_vecor_size_key = "dq_embedding_vecor_size"


class DQSessionManager:
    @staticmethod
    def is_app_state(app_state):
        return STSessionManager.get_value_for_key(DQSessionKeys.dq_app_state) == app_state

    # def append_to_file_list(file_info):
    #     file_list = STSessionManager.get_value_for_key(
    #         DQSessionKeys.dq_file_list)
    #     file_list.append(file_info)
    #     STSessionManager.set_key(DQSessionKeys.dq_file_list, file_list)


class DQPageManager:

    @staticmethod
    def get_value_for_key(session_key, is_raw=False):
        if is_raw:
            session_key = type('obj', (object,), {'name': session_key})
        # print(session_key.name)
        return STSessionManager.get_value_for_key(session_key=session_key)

    def set_key(session_key, value, is_raw=False):
        if is_raw:
            session_key = type('obj', (object,), {'name': session_key})
        # print(session_key.name)
        return STSessionManager.set_key(session_key=session_key, value=value)

    def initialize_page():
        STSessionManager.initialize_key(
            DQSessionKeys.dq_app_state, DQSettings.DQAppState.upload)
        STSessionManager.initialize_key(DQSessionKeys.dq_file_list, [])
        STSessionManager.initialize_key(
            DQSessionKeys.dq_vectordb_manager, None)
        STSessionManager.initialize_key(DQSessionKeys.dq_query_text, "")

    # def update_settings(dqui_key, value):
    #     if dqui_key == DQSettings.UIKeys.chunk_size_key:
    #         c_value = STSessionManager.get_value_for_key(
    #             DQSessionKeys.dq_embedding_settings)
    #         c_value['chunk_size'] = value
    #         STSessionManager.set_key(
    #             DQSessionKeys.dq_embedding_settings, c_value)
    #     if dqui_key == DQSettings.UIKeys.chunk_overlap_key:
    #         c_value = STSessionManager.get_value_for_key(
    #             DQSessionKeys.dq_embedding_settings)
    #         c_value['chunk_overlap'] = value
    #         STSessionManager.set_key(
    #             DQSessionKeys.dq_embedding_settings, c_value)
    #     if dqui_key == DQSettings.UIKeys.embedding_type_key:
    #         c_value = STSessionManager.get_value_for_key(
    #             DQSessionKeys.dq_embedding_settings)
    #         c_value['embedding_type'] = value
    #         STSessionManager.set_key(
    #             DQSessionKeys.dq_embedding_settings, c_value)

    def display_file_uploader():
        # print(DQSessionManager.is_app_state(DQSettings.DQAppState.upload))
        return DQSessionManager.is_app_state(DQSettings.DQAppState.upload)

    def update_file_info(uploaded_files):
        documents = []
        file_info_list = []
        for upoaded_file in uploaded_files:
            file_text = str(upoaded_file.read(), "utf-8")
            file_info = {'name': upoaded_file.name,
                         'size': upoaded_file.size, "type": upoaded_file.type}
            file_info_list.append(file_info)
            documents.append(DocumentUtil.document_from_text(
                text=file_text, info=file_info))
        # print(f"Uploaded files: {file_info_list}")
        STSessionManager.set_key(DQSessionKeys.dq_file_list, file_info_list)
        # print(documents)
        return documents

    def display_file_information():
        return STSessionManager.is_key_not_empty(DQSessionKeys.dq_file_list)

    def display_file_process_options():
        return DQPageManager.display_file_information() and DQPageManager.display_file_uploader()

    def create_db(documents):
        # Split document
        documents_chunked = DQPageManager.split_document(documents=documents)
        # get embegging model
        embedding_model = DQPageManager.get_embedding_model()
        print(documents_chunked)
        # embeddings = embedding_model.embed_documents(texts)
        # print(
        #     f"Check Embedding size: Total texts: {len(texts)} and Total Embddings: {len(embeddings)} with mebedding size {len(embeddings[0]) if len(embeddings)>0 else 0}")
        # # set total embeddings
        # DQPageManager.set_key(
        #     DQSettings.Info.embedding_total_key, value=len(texts), is_raw=True)
        # create vectordb
        DQPageManager.store_documents_in_db(
            embedding_model=embedding_model, documents=documents_chunked)
        # update app state
        STSessionManager.set_key(
            DQSessionKeys.dq_app_state, DQSettings.DQAppState.dbready)

    def split_document(documents):
        return DocumentUtil.split_documents_to_texts(documents=documents,
                                                     chunk_size=DQPageManager.get_value_for_key(
                                                         DQSettings.Embeddings.chunk_size_key, is_raw=True),
                                                     chunk_overlap_per=DQPageManager.get_value_for_key(
                                                         DQSettings.Embeddings.chunk_overlap_key, is_raw=True)
                                                     )

    def get_embedding_model(api_key=None):
        embedding_type = DQPageManager.get_value_for_key(
            DQSettings.Embeddings.embedding_type_key, is_raw=True)
        if embedding_type == EmbeddingType.OpenAI:
            api_key = DQPageManager.get_value_for_key(
                STSessionKeys.openai_api_key)
        embedding_model = EmbeddingManager.get_embedding_model(
            api_key=api_key, emb_model_type=embedding_type)
        return embedding_model

    def store_documents_in_db(embedding_model, documents):
        vectordb = DQPageManager.get_value_for_key(
            DQSessionKeys.dq_vectordb_manager)
        if not vectordb or not vectordb.vdb:
            persistent = DeploymentManager().is_dev_or_test()
            print(f"Creating vecotr DB (is persistent? {persistent})")
            db_dir = VectorDBManager.get_default_persistent_dir() if persistent else None
            vectordb = VectorDBManager(
                embedding_model=embedding_model, db_dir=db_dir, documents=documents)
            STSessionManager.set_key(
                DQSessionKeys.dq_vectordb_manager, vectordb)

        # print(vectordb.embedding_model)
        # print(vectordb.vdb)
        return vectordb

    def get_vdb_size():
        vectordb = DQPageManager.get_value_for_key(
            DQSessionKeys.dq_vectordb_manager)
        if not vectordb:
            return None
        else:
            return vectordb.get_db_size()

    def display_query():
        return DQSessionManager.is_app_state(DQSettings.DQAppState.dbready)

    def get_prompt():
        query = DQPageManager.get_value_for_key(DQSessionKeys.dq_query_text)
        return query

    def get_retriever():
        vectordb = DQPageManager.get_value_for_key(
            DQSessionKeys.dq_vectordb_manager)
        search_type = DQPageManager.get_value_for_key(
            DQSettings.Retriever.search_type_key, is_raw=True)
        max_match = DQPageManager.get_value_for_key(
            DQSettings.Retriever.max_match_key, is_raw=True)
        return vectordb.get_retriever(search_type=search_type, max_match=max_match)

    def get_chain(llm, retriever, chain_type="stuff"):
        cite_sources = DQPageManager.get_value_for_key(
            DQSettings.LLMResponse.cite_source_key, is_raw=True)
        # print(f"LLM: {llm}")
        # print(f"Retrieve: {retriever}")
        # print(f"cite surces: {cite_sources}")
        # create the chain to answer questions
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type=chain_type,
                                               retriever=retriever,
                                               return_source_documents=cite_sources)
        return qa_chain

    # def process_llm_response(llm_response, cite_sources=True):
    #     if cite_sources:
    #         print('\n\nSources:')
    #         ind = 1
    #         for source in llm_response["source_documents"]:
    #             print(f"\n{ind}. {source.metadata['source']} ************")
    #             print(source.page_content)
    #             ind += 1
    #     print(f"\n************\nQuery: {llm_response['query']}")
    #     print(f"\nResponse: {llm_response['result']}")

    def get_llm_response():
        # print(f"entering get llm response......................")
        prompt = DQPageManager.get_prompt()
        if not prompt:
            return None, "There is no query", None

        # check if api_key  is set
        if not STSessionManager.is_api_key_set():
            response = "API Key not set. LLM not run"
            return prompt, response, None

        # check if run llm is set
        if not STSessionManager.llm_ready_to_run():
            response = "I am not real LLM. Check the \" Run LLM\" option to rum LLM"
            return prompt, response, None

        if not STSessionManager.is_llm_manager_set() or not STSessionManager.get_llm_manager().llm:
            # print(STSessionManager.get_api_key())
            (STSessionManager.get_api_key())
            llm = LLMManager.get_llm(STSessionManager.get_api_key())
            # print(f"created LLM: {llm}")
            # print(llm)
            llm_manager = LLMManager(llm)
            # print(f"Seting LLM manager key")
            STSessionManager.set_llm_manager(llm_manager)

        llm_manager = STSessionManager.get_llm_manager()
        # print(llm_manager)
        llm = llm_manager.llm
        retriever = DQPageManager.get_retriever()
        qa_chain = DQPageManager.get_chain(llm=llm, retriever=retriever)
        # print(f"QA chain called: {qa_chain}......................")
        try:
            qa_response = qa_chain(prompt)
        except Exception as e:
            qa_response = e

        # print("Response from chain: ........")
        # print(qa_response)
        response = qa_response['result']
        # cite_sources = DQPageManager.get_value_for_key(
        #     DQSettings.LLMResponse.cite_source_key, is_raw=True)
        if "source_documents" in qa_response:
            sources = qa_response["source_documents"]
        else:
            sources = None

        return prompt, response, sources

    def start_over(clear_db=False):
        # reset app state
        STSessionManager.set_key(
            DQSessionKeys.dq_app_state, DQSettings.DQAppState.upload)
        # clear vector DB
        vectordb_manager = DQPageManager.get_value_for_key(
            DQSessionKeys.dq_vectordb_manager)
        if clear_db and vectordb_manager:
            vectordb_manager.vdb.delete_collection()

    # def store_docs_in_db(file_path=None, type=None, api_key=None, emb_model="huggingface", chunk_size=1000, chunk_overlap_per=20):
    #     docs = DocumentUtil.get_documents(type=type)
    #     texts = DocumentUtil.get_texts(
    #         documents=docs, chunk_size=chunk_size, chunk_overlap_per=chunk_overlap_per)
    #     # print(len(texts))
    #     embedding = DocumentUtil.get_embedding_model(
    #         api_key=api_key, emb_model=emb_model)
    #     # print(len(embeddings))
    #     # print(texts)
    #     DocumentUtil.store_in_vector_store(
    #         texts=texts, embedding=embedding)

    # def test_hf_llm(api_token):
    #     question = "What is the best flour for pasta? "
    #     template = """Question: {question}
    #     Answer: Let's think step by step."""
    #     prompt = PromptTemplate(
    #         template=template, input_variables=["question"])

    #     # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    #     llm = LLMManager.get_llm(api_token)
    #     llm_chain = LLMChain(prompt=prompt, llm=llm)
    #     answer = llm_chain.run(question)
    #     print(answer)
    #     return (answer)

    # def test(llm, api_key, emb_model="huggingface", is_write=True, is_reset=False, run_prod=True):
    #     search_type = "similarity"  # mmr, similarity_score_threshold
    #     embedding = EmbeddingManager.get_embedding_model(
    #         api_key=api_key, emb_model=emb_model)
    #     vectordb = VectorDBManager.get_vector_store(embedding=embedding)
    #     if is_reset:
    #         # print("Deleting Chroma colection: langchain")
    #         # vectordb.delete_collection()
    #         # print("Deleted Chroma colection: langchain")
    #         VectorDBManager.delete_vdb_dir()
    #     if is_write:
    #         documents = DocumentUtil.get_documents(type="dir")
    #         texts = DocumentUtil.get_texts(documents, chunk_size=1000)
    #         VectorDBManager.store_in_vector_store(
    #             texts=texts, embedding=embedding)
    #     # print(texts)
    #     query = "Which floor did Sue and Johnsy live?"
    #     query = "What is the size of the letter-box in the hall below?"
    #     response = DQPageManager.get_llm_response(llm=llm,
    #                                               query=query, vectoredb=vectordb, api_key=api_key, search_type=search_type, max_count=3)

    #     # print(response)

    #     DocumentUtil.process_llm_response(response, True)


def main():
    return


if __name__ == "__main__":
    main()
