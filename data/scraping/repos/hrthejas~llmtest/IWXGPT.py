import os
import gradio as gr
from getpass import getpass
from llmtest import constants, vectorstore, ingest, embeddings, indextype
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from llmtest.IWXRetriever import IWXRetriever
from llmtest.MysqlLogger import MysqlLogger
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain


class IWXGPT:
    doc_vector_stores = []
    api_vector_stores = []
    api_prompt = None
    doc_prompt = None
    code_prompt = None
    summary_prompt = None
    api_help_prompt = None
    llm_model = None
    vector_embeddings = None
    api_iwx_retriever = None
    doc_iwx_retriever = None
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.clear()
    app_args = ["model_name", "temperature", "index_base_path", "docs_index_name_prefix", "api_index_name_prefix",
                "max_new_tokens", "mount_gdrive", "gdrive_mount_base_bath", "embedding_model_name",
                "api_prompt_template", "doc_prompt_template", "code_prompt_template", "summary_prompt_template"]

    model_name = constants.OPEN_AI_MODEL_NAME
    temperature = constants.OPEN_AI_TEMP
    max_new_tokens = constants.MAX_NEW_TOKENS
    docs_base_path = constants.DOCS_BASE_PATH
    index_base_path = constants.OAI_INDEX_BASE_PATH
    docs_index_name_prefix = constants.DOC_INDEX_NAME_PREFIX
    api_index_name_prefix = constants.API_INDEX_NAME_PREFIX
    mount_gdrive = True
    gdrive_mount_base_bath = constants.GDRIVE_MOUNT_BASE_PATH
    embedding_model_name = None
    api_prompt_template = constants.API_QUESTION_PROMPT
    doc_prompt_template = constants.DOC_QUESTION_PROMPT
    code_prompt_template = constants.DEFAULT_PROMPT_FOR_CODE
    summary_prompt_template = constants.DEFAULT_PROMPT_FOR_SUMMARY
    api_help_prompt_template = constants.DEFAULT_PROMPT_FOR_API_HELP

    def __getitem__(self, item):
        return item

    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            valid_kwargs = {name: kwargs.pop(name) for name in self.app_args if name in kwargs}
            for key, value in valid_kwargs.items():
                if hasattr(self, key):
                    print("Setting attribute " + key)
                    setattr(self, key, value)

        if self.mount_gdrive:
            ingest.mountGoogleDrive(self.gdrive_mount_base_bath)

        os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")

        pass

    def initialize_chat(self):
        self.vector_embeddings = embeddings.get_openai_embeddings()
        print("Got the embeddigns")

        self.llm_model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature,
                                    max_tokens=self.max_new_tokens)
        print("Loaded LLM Model")

        for prefix in self.docs_index_name_prefix:
            self.doc_vector_stores.append(vectorstore.get_vector_store(index_base_path=self.index_base_path,
                                                                       index_name_prefix=prefix,
                                                                       docs_base_path=self.docs_base_path,
                                                                       embeddings=self.vector_embeddings))
            print("Loaded vector store from " + self.index_base_path + "/" + prefix)
        self.doc_iwx_retriever = IWXRetriever()
        self.doc_iwx_retriever.initialise(self.doc_vector_stores)

        for prefix in self.api_index_name_prefix:
            self.api_vector_stores.append(vectorstore.get_vector_store(index_base_path=self.index_base_path,
                                                                       index_name_prefix=prefix,
                                                                       docs_base_path=self.docs_base_path,
                                                                       embeddings=self.vector_embeddings))
            print("Loaded vector store from " + self.index_base_path + "/" + prefix)
        self.api_iwx_retriever = IWXRetriever()
        self.api_iwx_retriever.initialise(self.api_vector_stores)

        self.api_prompt = PromptTemplate(template=self.api_prompt_template,
                                         input_variables=["context", "question"])

        self.doc_prompt = PromptTemplate(template=self.doc_prompt_template,
                                         input_variables=["context", "question"])

        self.code_prompt = PromptTemplate(template=self.code_prompt_template,
                                          input_variables=["context", "question"])

        self.summary_prompt = PromptTemplate(template=self.summary_prompt_template,
                                             input_variables=["context", "question"])

        self.api_help_prompt = PromptTemplate(template=self.api_help_prompt_template,
                                              input_variables=["context", "question"])

        print("Loaded all prompts")
        print("Init complete")
        pass

    def ask(self, answer_type, query, similarity_search_k=4, api_prompt=None,
            doc_prompt=None, code_prompt=None, summary_prompt=None, api_help_prompt=None, clear_memory=False):
        from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

        self.api_iwx_retriever.set_search_k(similarity_search_k)
        self.doc_iwx_retriever.set_search_k(similarity_search_k)

        if clear_memory:
            self.memory.clear()

        if api_prompt is None:
            api_prompt = self.api_prompt
        if doc_prompt is None:
            doc_prompt = self.doc_prompt
        if code_prompt is None:
            code_prompt = self.code_prompt
        if summary_prompt is None:
            summary_prompt = self.summary_prompt
        if api_help_prompt is None:
            api_help_prompt = self.api_help_prompt

        if self.llm_model is not None:
            chain = None
            if answer_type == "Summary":
                search_results = ingest.get_doc_from_text(query)
                local_qa_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=summary_prompt)
                result = local_qa_chain({"input_documents": search_results, "question": query})
                bot_message = result["output_text"]
            else:
                if answer_type == "API":

                    chain = ConversationalRetrievalChain.from_llm(self.llm_model, memory=self.memory,
                                                                  retriever=self.api_iwx_retriever,
                                                                  combine_docs_chain_kwargs={"prompt": api_prompt})

                    # question_generator = LLMChain(llm=self.llm_model, prompt=CONDENSE_QUESTION_PROMPT)
                    # combine_docs_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=api_prompt)
                    # chain = ConversationalRetrievalChain(retriever=self.api_iwx_retriever,
                    #                                      question_generator=question_generator,
                    #                                      combine_docs_chain=combine_docs_chain)
                elif answer_type == "API_HELP":
                    question_generator = LLMChain(llm=self.llm_model, prompt=CONDENSE_QUESTION_PROMPT)
                    combine_docs_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=api_help_prompt)
                    chain = ConversationalRetrievalChain(retriever=self.api_iwx_retriever,
                                                         question_generator=question_generator,
                                                         combine_docs_chain=combine_docs_chain)
                elif answer_type == "Code":
                    question_generator = LLMChain(llm=self.llm_model, prompt=CONDENSE_QUESTION_PROMPT)
                    combine_docs_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=code_prompt)
                    chain = ConversationalRetrievalChain(retriever=self.api_iwx_retriever,
                                                         question_generator=question_generator,
                                                         combine_docs_chain=combine_docs_chain)
                elif answer_type == "Doc":
                    question_generator = LLMChain(llm=self.llm_model, prompt=CONDENSE_QUESTION_PROMPT)
                    combine_docs_chain = load_qa_chain(llm=self.llm_model, chain_type="stuff", prompt=doc_prompt)
                    chain = ConversationalRetrievalChain(retriever=self.doc_iwx_retriever,
                                                         question_generator=question_generator,
                                                         combine_docs_chain=combine_docs_chain)
                else:
                    raise Exception("Unknown Answer Type")
            if chain is not None:
                result = chain({"question": query})
                bot_message = result['answer']
            else:
                bot_message = "Chain is none"
        else:
            bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"

        print("Answer")
        print(bot_message)
        return bot_message

    def ask_with_prompt(self, answer_type, query, similarity_search_k=4,
                        api_prompt_template=api_prompt_template,
                        doc_prompt_template=doc_prompt_template,
                        code_prompt_template=code_prompt_template,
                        summary_prompt_template=summary_prompt_template,
                        api_help_prompt_template=api_help_prompt_template):

        api_prompt = PromptTemplate(template=api_prompt_template,
                                    input_variables=["context", "question"])

        doc_prompt = PromptTemplate(template=doc_prompt_template,
                                    input_variables=["context", "question"])

        code_prompt = PromptTemplate(template=code_prompt_template,
                                     input_variables=["context", "question"])

        summary_prompt = PromptTemplate(template=summary_prompt_template,
                                        input_variables=["context", "question"])

        api_help_prompt = PromptTemplate(template=api_help_prompt_template,
                                         input_variables=["context", "question"])

        return self.ask(answer_type, query, similarity_search_k, api_prompt, doc_prompt, code_prompt, summary_prompt,
                        api_help_prompt, False)

    def start_chat(self, debug=True, use_queue=False, share_ui=True, similarity_search_k=4, record_feedback=True,
                   api_prompt_template=constants.API_QUESTION_PROMPT,
                   doc_prompt_template=constants.DOC_QUESTION_PROMPT,
                   code_prompt_template=constants.DEFAULT_PROMPT_FOR_CODE,
                   summary_prompt_template=constants.DEFAULT_PROMPT_FOR_SUMMARY,
                   api_help_prompt_template=api_help_prompt_template,
                   add_summary_answer_type=False):

        if add_summary_answer_type:
            choices = ['API', 'Docs', 'Code', 'Summary']
        else:
            choices = ['API', 'Docs', 'Code']
        data = [('Bad', '1'), ('Ok', '2'), ('Good', '3'), ('Very Good', '4'), ('Perfect', '5')]

        def chatbot(choice_selected, message):
            return self.ask_with_prompt(choice_selected, message, similarity_search_k=similarity_search_k,
                                        api_prompt_template=api_prompt_template,
                                        doc_prompt_template=doc_prompt_template,
                                        code_prompt_template=code_prompt_template,
                                        summary_prompt_template=summary_prompt_template,
                                        api_help_prompt_template=api_help_prompt_template)

        msg = gr.Textbox(label="User Question")
        submit = gr.Button("Submit")
        choice = gr.inputs.Dropdown(choices=choices, default="Docs", label="Choose question Type")
        output_textbox = gr.outputs.Textbox(label="IWX Bot")
        output_textbox.show_copy_button = True
        output_textbox.lines = 10
        output_textbox.max_lines = 10

        # output_textbox1 = gr.outputs.Textbox(label="Reference Docs")
        # output_textbox1.lines = 2
        # output_textbox1.max_lines = 2

        if record_feedback:
            interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox],
                                     theme="huggingface",
                                     title="IWX CHATBOT", allow_flagging="manual", flagging_callback=MysqlLogger(),
                                     flagging_options=data)
        else:
            interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=[output_textbox],
                                     theme="huggingface",
                                     title="IWX CHATBOT", allow_flagging="never")
        if use_queue:
            interface.queue().launch(debug=debug, share=share_ui)
        else:
            interface.launch(debug=debug, share=share_ui)

    def overwrite_vector_store(self, docs_type, docs_base_path, index_base_path, index_name,
                               chunk_size, chunk_overlap, embeddings,
                               vector_store_type=indextype.IndexType.FAISS_INDEX, model_name="gpt-3.5-turbo",
                               encoding_name="cl100k_base"):
        if docs_type == "CSV":
            data = ingest.get_csv_docs_tiktoken(docs_base_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                model_name=model_name, encoding_name=encoding_name)
        elif docs_type == "MD":
            data = ingest.getMarkDownDocs(docs_base_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif docs_type == "HTML":
            data = ingest.getHTMLDocs(docs_base_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif docs_type == "JSON":
            data = ingest.get_json_docs(docs_base_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise Exception("Un supported doc type " + docs_type)

        store = vectorstore.get_vector_store_from_docs(data, index_base_path=index_base_path,
                                                       index_name_prefix=index_name,
                                                       embeddings=embeddings, index_type=vector_store_type,
                                                       is_overwrite=True)
        if store is not None:
            print(
                "Created vector store and persisted at " + index_base_path + "/" + vector_store_type.name + "/" + index_name)
        else:
            raise Exception("Error while creating or persisting vector store")

        pass

    def get_embeddings(self):
        return embeddings.get_openai_embeddings()
