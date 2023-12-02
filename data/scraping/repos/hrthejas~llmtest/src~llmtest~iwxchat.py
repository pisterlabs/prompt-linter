import gradio as gr
import time
import os
from getpass import getpass

import torch
from gradio import FlaggingCallback
from gradio.components import IOComponent
from typing import Any
from transformers import pipeline

from llmtest import constants, startchat, ingest, storage, embeddings, vectorstore, llmloader, utils

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings
)


def get_question(user_input, use_prompt, prompt, choice_selected):
    if use_prompt and choice_selected == "API":
        return prompt + '\n' + user_input
    else:
        return user_input


def record_answers(query, open_ai_answer, local_model_answer):
    try:
        storage.insert_data(constants.USER_NAME, query, open_ai_answer, local_model_answer)
    except:
        print("Error while recording answers to db")
        pass


def get_openai_qa_chain(open_ai_llm, api_index_name_prefix, docs_base_path, docs_index_name_prefix, index_base_path,
                        search_kwargs,
                        search_type, is_openai_model):
    openai_embeddings = embeddings.get_openai_embeddings()

    docs_retriever, api_retriever = get_retrievers(openai_embeddings, docs_base_path, index_base_path,
                                                   api_index_name_prefix, docs_index_name_prefix, search_kwargs,
                                                   search_type, is_openai_model)

    openai_docs_qa_chain = startchat.get_openai_model_qa_chain(open_ai_llm, docs_retriever)
    openai_api_qa_chain = startchat.get_openai_model_qa_chain(open_ai_llm, api_retriever)

    return openai_docs_qa_chain, openai_api_qa_chain


def get_local_qa_chain(llm, embedding_class, model_name, api_index_name_prefix, docs_base_path, docs_index_name_prefix,
                       index_base_path, search_kwargs,
                       search_type, is_openai_model):
    hf_embeddings = embeddings.get_embeddings(embedding_class, model_name)

    docs_retriever, api_retriever = get_retrievers(hf_embeddings, docs_base_path, index_base_path,
                                                   api_index_name_prefix, docs_index_name_prefix, search_kwargs,
                                                   search_type, is_openai_model)

    openai_docs_qa_chain = startchat.get_openai_model_qa_chain(llm, docs_retriever)
    openai_api_qa_chain = startchat.get_openai_model_qa_chain(llm, api_retriever)

    return openai_docs_qa_chain, openai_api_qa_chain


def get_retrievers(model_embeddings, docs_base_path, index_base_path, api_index_name_prefix, docs_index_name_prefix,
                   search_kwargs,
                   search_type, is_openai_model):
    doc_vector_store, api_vector_store = get_vector_stores(model_embeddings, docs_base_path, index_base_path,
                                                           api_index_name_prefix, docs_index_name_prefix,
                                                           is_openai_model)

    docs_retriever = vectorstore.get_retriever_from_store(doc_vector_store, search_type=search_type,
                                                          search_kwargs=search_kwargs)

    api_retriever = vectorstore.get_retriever_from_store(api_vector_store, search_type=search_type,
                                                         search_kwargs=search_kwargs)
    return docs_retriever, api_retriever


class MysqlLogger(FlaggingCallback):

    def __init__(self):
        pass

    def setup(self, components: list[IOComponent], flagging_dir: str = None):
        self.components = components
        self.flagging_dir = flagging_dir
        print("here in setup")

    def flag(
            self,
            flag_data: list[Any],
            flag_option: str = "",
            username: str = None,
    ) -> int:
        data = []
        for component, sample in zip(self.components, flag_data):
            data.append(
                component.deserialize(
                    sample,
                    None,
                    None,
                )
            )
        data.append(flag_option)
        if len(data[1]) > 0 and len(data[2]) > 0:
            storage.insert_with_rating(constants.USER_NAME, data[0], data[1], data[2], data[3], data[4])
        else:
            print("no data to log")

        return 1


def get_vector_stores(model_embeddings, docs_base_path, index_base_path, api_index_name_prefix, docs_index_name_prefix,
                      is_openai_model):
    if is_openai_model:
        index_base_path = index_base_path + "/openai/"
    else:
        index_base_path = index_base_path + "/hf/"
    doc_vector_store = vectorstore.get_vector_store(index_base_path=index_base_path,
                                                    index_name_prefix=docs_index_name_prefix,
                                                    docs_base_path=docs_base_path, embeddings=model_embeddings)
    api_vector_store = vectorstore.get_vector_store(index_base_path=index_base_path,
                                                    index_name_prefix=api_index_name_prefix,
                                                    docs_base_path=docs_base_path, embeddings=model_embeddings)
    return doc_vector_store, api_vector_store


def start_iwx_only_chat(local_model_id=constants.DEFAULT_MODEL_NAME,
                        docs_base_path=constants.DOCS_BASE_PATH, index_base_path=constants.INDEX_BASE_PATH,
                        docs_index_name_prefix=constants.DOC_INDEX_NAME_PREFIX,
                        api_index_name_prefix=constants.API_INDEX_NAME_PREFIX,
                        max_new_tokens=constants.MAX_NEW_TOKENS, use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
                        set_device_map=constants.SET_DEVICE_MAP,
                        mount_gdrive=True,
                        share_chat_ui=True, debug=False, gdrive_mount_base_bath=constants.GDRIVE_MOUNT_BASE_PATH,
                        device_map=constants.DEFAULT_DEVICE_MAP, use_simple_llm_loader=False,
                        embedding_class=HuggingFaceInstructEmbeddings, model_name="hkunlp/instructor-large",
                        use_queue=True, is_gptq_model=False, custom_quantization_config=None, use_safetensors=False,
                        use_triton=False, set_torch_dtype=False, torch_dtype=torch.bfloat16):
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate

    api_prompt_template = constants.API_QUESTION_PROMPT
    api_prompt = PromptTemplate(template=api_prompt_template,
                                input_variables=["context", "question"])
    doc_prompt_template = constants.DOC_QUESTION_PROMPT
    doc_prompt = PromptTemplate(template=doc_prompt_template,
                                input_variables=["context", "question"])

    if mount_gdrive:
        ingest.mountGoogleDrive(mount_location=gdrive_mount_base_bath)

    llm = llmloader.load_llm(local_model_id, use_4bit_quantization=use_4bit_quantization, set_device_map=set_device_map,
                             max_new_tokens=max_new_tokens, device_map=device_map,
                             use_simple_llm_loader=use_simple_llm_loader, is_quantized_gptq_model=is_gptq_model,
                             custom_quantiztion_config=custom_quantization_config, use_triton=use_triton,
                             use_safetensors=use_safetensors, set_torch_dtype=set_torch_dtype, torch_dtype=torch_dtype)

    hf_embeddings = embeddings.get_embeddings(embedding_class, model_name)

    index_base_path = index_base_path + "/hf/"

    local_doc_vector_stores = []
    for prefix in docs_index_name_prefix:
        local_doc_vector_stores.append(vectorstore.get_vector_store(index_base_path=index_base_path,
                                                                    index_name_prefix=prefix,
                                                                    docs_base_path=docs_base_path,
                                                                    embeddings=hf_embeddings))

    local_api_vector_stores = []
    for prefix in api_index_name_prefix:
        local_api_vector_stores.append(vectorstore.get_vector_store(index_base_path=index_base_path,
                                                                   index_name_prefix=prefix,
                                                                   docs_base_path=docs_base_path,
                                                                   embeddings=hf_embeddings))

    choices = ['Docs', 'API']
    data = [('Bad', '1'), ('Ok', '2'), ('Good', '3'), ('Very Good', '4'), ('Perfect', '5')]

    def chatbot(choice_selected, message):
        query = message
        reference_docs = ""
        if llm is not None:
            search_results = None
            local_qa_chain = None
            if choice_selected == "API":
                for api_vector_store in local_api_vector_stores:
                    if search_results is None:
                        search_results = api_vector_store.similarity_search(query)
                    else:
                        search_results = search_results + api_vector_store.similarity_search(query)
                local_qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=api_prompt)
            else:
                for doc_vector_store in local_doc_vector_stores:
                    if search_results is None:
                        search_results = doc_vector_store.similarity_search(query)
                    else:
                        search_results = search_results + doc_vector_store.similarity_search(query)
                local_qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=doc_prompt)

            if local_qa_chain is not None and search_results is not None:
                result = local_qa_chain({"input_documents": search_results, "question": query})
                bot_message = result["output_text"]
                for doc in search_results:
                    reference_docs = reference_docs + '\n' + str(doc.metadata.get('source'))
            else:
                bot_message = "No matching docs found on the vector store"
        else:
            bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
        # record_answers(query, "OPen AI Not configured", bot_message)
        print(bot_message)
        print(reference_docs)
        return bot_message, reference_docs

    msg = gr.Textbox(label="User Question")
    submit = gr.Button("Submit")
    choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
    output_textbox = gr.outputs.Textbox(label="IWX Bot")
    output_textbox.show_copy_button = True
    output_textbox.lines = 10
    output_textbox.max_lines = 10

    output_textbox1 = gr.outputs.Textbox(label="Reference Docs")
    output_textbox1.lines = 2
    output_textbox1.max_lines = 2

    interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox, output_textbox1],
                             theme="gradio/monochrome",
                             title="IWX CHATBOT", allow_flagging="manual", flagging_callback=MysqlLogger(),
                             flagging_options=data)
    if use_queue:
        interface.queue()
    interface.launch(debug=debug, share=share_chat_ui)


def query_llm(llm, api_prompt, doc_prompt, api_vector_stores, doc_vector_stores, answer_type, query,similarity_search_k=4):
    from langchain.chains.question_answering import load_qa_chain
    reference_docs = ""
    if llm is not None:
        search_results = None
        local_qa_chain = None
        if answer_type == "API":
            for api_vector_store in api_vector_stores:
                if search_results is None:
                    search_results = api_vector_store.similarity_search(query,k=similarity_search_k)
                else:
                    search_results = search_results + api_vector_store.similarity_search(query,k=similarity_search_k)
            local_qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=api_prompt)
        else:
            for doc_vector_store in doc_vector_stores:
                if search_results is None:
                    search_results = doc_vector_store.similarity_search(query,k=similarity_search_k)
                else:
                    search_results = search_results + doc_vector_store.similarity_search(queryk=similarity_search_k)
            local_qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=doc_prompt)

        if local_qa_chain is not None and search_results is not None:
            result = local_qa_chain({"input_documents": search_results, "question": query})
            bot_message = result["output_text"]
            for doc in search_results:
                reference_docs = reference_docs + '\n' + str(doc.metadata.get('source'))
        else:
            bot_message = "No matching docs found on the vector store"
    else:
        bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
    return bot_message, reference_docs
