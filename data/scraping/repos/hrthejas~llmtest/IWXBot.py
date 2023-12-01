import torch
import gradio as gr
from llmtest import llmloader, constants, vectorstore, ingest, embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings
)
from langchain.chat_models import ChatOpenAI

from llmtest.IWXRetriever import IWXRetriever
from llmtest.MysqlLogger import MysqlLogger
from langchain.chains import LLMChain
import os
from getpass import getpass
from langchain.chains import RetrievalQA
from langchain.chains.conversation.base import ConversationChain
from langchain.text_splitter import CharacterTextSplitter


class IWXBot:
    doc_vector_stores = []
    api_vector_stores = []
    api_prompt = None
    doc_prompt = None
    code_prompt = None
    summary_prompt = None
    api_help_prompt = None
    llm_model = None
    summary_llm_model = None
    vector_embeddings = None
    api_iwx_retriever = None
    doc_iwx_retriever = None
    chat_history = []
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    app_args = ["model_id", "docs_base_path", "index_base_path", "docs_index_name_prefix", "api_index_name_prefix",
                "max_new_tokens", "use_4bit_quantization", "set_device_map", "mount_gdrive", "gdrive_mount_base_bath",
                "device_map", "use_simple_llm_loader", "embedding_class", "model_name", "is_gptq_model",
                "custom_quantization_config", "use_safetensors", "use_triton", "set_torch_dtype", "torch_dtype",
                "api_prompt_template", "doc_prompt_template", "code_prompt_template", "model_basename", "iwx_base_url"]

    model_id = constants.DEFAULT_MODEL_NAME
    docs_base_path = constants.DOCS_BASE_PATH
    index_base_path = constants.HF_INDEX_BASE_PATH
    docs_index_name_prefix = constants.DOC_INDEX_NAME_PREFIX
    api_index_name_prefix = constants.API_INDEX_NAME_PREFIX
    max_new_tokens = constants.MAX_NEW_TOKENS
    use_4bit_quantization = constants.USE_4_BIT_QUANTIZATION
    set_device_map = constants.SET_DEVICE_MAP
    mount_gdrive = True
    gdrive_mount_base_bath = constants.GDRIVE_MOUNT_BASE_PATH
    device_map = constants.DEFAULT_DEVICE_MAP
    use_simple_llm_loader = False
    embedding_class = HuggingFaceInstructEmbeddings
    model_name = "hkunlp/instructor-large"
    is_gptq_model = False
    custom_quantization_config = None
    use_safetensors = False
    use_triton = False
    set_torch_dtype = False
    torch_dtype = torch.bfloat16
    api_prompt_template = constants.API_QUESTION_PROMPT
    doc_prompt_template = constants.DOC_QUESTION_PROMPT
    code_prompt_template = constants.DEFAULT_PROMPT_FOR_CODE
    summary_prompt_template = constants.DEFAULT_PROMPT_FOR_SUMMARY
    api_help_prompt_template = constants.DEFAULT_PROMPT_FOR_API_HELP
    model_basename = None
    iwx_base_url = 'http://10.37.0.7:3000'

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

        self.vector_embeddings = embeddings.get_embeddings(self.embedding_class, self.model_name)

        for prefix in self.docs_index_name_prefix:
            self.doc_vector_stores.append(vectorstore.get_vector_store(index_base_path=self.index_base_path,
                                                                       index_name_prefix=prefix,
                                                                       docs_base_path=self.docs_base_path,
                                                                       embeddings=self.vector_embeddings))
        self.doc_iwx_retriever = IWXRetriever()
        self.doc_iwx_retriever.initialise(self.doc_vector_stores)

        for prefix in self.api_index_name_prefix:
            self.api_vector_stores.append(vectorstore.get_vector_store(index_base_path=self.index_base_path,
                                                                       index_name_prefix=prefix,
                                                                       docs_base_path=self.docs_base_path,
                                                                       embeddings=self.vector_embeddings))
        self.api_iwx_retriever = IWXRetriever()
        self.api_iwx_retriever.initialise(self.api_vector_stores)

        self.api_prompt = PromptTemplate(template=self.api_prompt_template,
                                         input_variables=["context", "question", "base_url"])

        self.doc_prompt = PromptTemplate(template=self.doc_prompt_template,
                                         input_variables=["context", "question", "base_url"])

        self.code_prompt = PromptTemplate(template=self.code_prompt_template,
                                          input_variables=["context", "question", "base_url"])

        self.summary_prompt = PromptTemplate(template=self.summary_prompt_template,
                                             input_variables=["context", "question", "base_url"])

        self.api_help_prompt = PromptTemplate(template=self.api_help_prompt_template,
                                              input_variables=["context", "question", "base_url"])

        print("Loaded all prompts")

        self.llm_model = llmloader.load_llm(self.model_id, use_4bit_quantization=self.use_4bit_quantization,
                                            set_device_map=self.set_device_map,
                                            max_new_tokens=self.max_new_tokens, device_map=self.device_map,
                                            use_simple_llm_loader=self.use_simple_llm_loader,
                                            is_quantized_gptq_model=self.is_gptq_model,
                                            custom_quantiztion_config=self.custom_quantization_config,
                                            use_triton=self.use_triton,
                                            use_safetensors=self.use_safetensors, set_torch_dtype=self.set_torch_dtype,
                                            torch_dtype=self.torch_dtype,
                                            model_basename=self.model_basename)
        print("Loaded base LLM")
        self.summary_llm_model = ChatOpenAI(model_name=constants.OPEN_AI_MODEL_NAME, temperature=constants.OPEN_AI_TEMP,
                                            max_tokens=self.max_new_tokens)
        print("Loaded Summary LLM")
        print("Init complete")
        pass

    def ask_with_memory(self, answer_type, query, similarity_search_k=2, api_prompt=None,
                        doc_prompt=None, code_prompt=None, summary_prompt=None, api_help_prompt=None,
                        new_chat=False):

        self.api_iwx_retriever.set_search_k(similarity_search_k)
        self.doc_iwx_retriever.set_search_k(similarity_search_k)

        if new_chat:
            self.chat_history = []

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
                local_qa_chain = load_qa_chain(llm=self.summary_llm_model, chain_type="stuff", prompt=summary_prompt)
                result = local_qa_chain(
                    {"input_documents": search_results, "question": query, "base_url": self.iwx_base_url})
                bot_message = result["output_text"]
            elif answer_type == "General":
                response = self.summary_llm_model.predict(query)
                # splitter = CharacterTextSplitter(chunk_size=512)
                # retriever = RetrievalQA(llm=self.summary_llm_model, splitter=splitter, top_k=1)
                # # Initialize the conversation chain
                # conversation_chain = ConversationChain(llm=self.summary_llm_model, retriever=retriever)
                # response = conversation_chain(query)
                # #
                # # gen_chain = RetrievalQA.from_chain_type(llm=self.summary_llm_model, chain_type="stuff")
                # # result = gen_chain(query)
                bot_message = response
            else:
                if answer_type == "API":
                    chain = ConversationalRetrievalChain.from_llm(self.llm_model,
                                                                  retriever=self.api_iwx_retriever,
                                                                  combine_docs_chain_kwargs={"prompt": api_prompt})
                elif answer_type == "API HELP":
                    chain = ConversationalRetrievalChain.from_llm(self.llm_model,
                                                                  retriever=self.api_iwx_retriever,
                                                                  combine_docs_chain_kwargs={"prompt": api_help_prompt})
                elif answer_type == "Code":
                    chain = ConversationalRetrievalChain.from_llm(self.llm_model,
                                                                  retriever=self.api_iwx_retriever,
                                                                  combine_docs_chain_kwargs={"prompt": code_prompt})
                elif answer_type == "Docs":
                    chain = ConversationalRetrievalChain.from_llm(self.llm_model,
                                                                  retriever=self.doc_iwx_retriever,
                                                                  combine_docs_chain_kwargs={"prompt": doc_prompt})
                else:
                    raise Exception("Unknown Answer Type")

                if chain is not None:
                    print(self.chat_history)
                    result = chain(
                        {"question": query, "chat_history": self.chat_history, "base_url": self.iwx_base_url})
                    bot_message = result['answer']
                else:
                    bot_message = "Chain is none"
        else:
            bot_message = "Seams like iwxchat model is not loaded or not requested to give answer"
        print(bot_message)
        self.chat_history = [(query, bot_message)]
        return bot_message

    def ask_with_prompt(self, answer_type, query, similarity_search_k=4,
                        api_prompt_template=api_prompt_template,
                        doc_prompt_template=doc_prompt_template,
                        code_prompt_template=code_prompt_template,
                        summary_prompt_template=summary_prompt_template,
                        api_help_prompt_template=api_help_prompt_template,
                        new_chat=False):

        api_prompt = PromptTemplate(template=api_prompt_template,
                                    input_variables=["context", "question", "base_url"])

        doc_prompt = PromptTemplate(template=doc_prompt_template,
                                    input_variables=["context", "question", "base_url"])

        code_prompt = PromptTemplate(template=code_prompt_template,
                                     input_variables=["context", "question", "base_url"])

        summary_prompt = PromptTemplate(template=summary_prompt_template,
                                        input_variables=["context", "question", "base_url"])

        api_help_prompt = PromptTemplate(template=api_help_prompt_template,
                                         input_variables=["context", "question", "base_url"])

        return self.ask_with_memory(answer_type, query, similarity_search_k, api_prompt, doc_prompt, code_prompt,
                                    summary_prompt, api_help_prompt, new_chat)

    def start_iwx_chat(self, debug=True, use_queue=False, share_ui=True, similarity_search_k=2, record_feedback=False,
                       add_summary_answer_type=True, api_prompt_template=constants.API_QUESTION_PROMPT,
                       doc_prompt_template=constants.DOC_QUESTION_PROMPT,
                       code_prompt_template=constants.DEFAULT_PROMPT_FOR_CODE,
                       summary_prompt_template=constants.DEFAULT_PROMPT_FOR_SUMMARY):

        if add_summary_answer_type:
            choices = ['API', 'Docs', 'Code', 'API HELP', 'Summary']
        else:
            choices = ['API', 'Docs', 'Code', 'API HELP']
        data = [('Bad', '1'), ('Ok', '2'), ('Good', '3'), ('Very Good', '4'), ('Perfect', '5')]

        def chatbot(choice_selected, message):
            return self.ask_with_prompt(choice_selected, message, similarity_search_k=similarity_search_k,
                                        new_chat=True, api_prompt_template=api_prompt_template,
                                        doc_prompt_template=doc_prompt_template,
                                        code_prompt_template=code_prompt_template,
                                        summary_prompt_template=summary_prompt_template)

        msg = gr.Textbox(label="User Question")
        submit = gr.Button("Submit")
        choice = gr.inputs.Dropdown(choices=choices, default="Code", label="Choose question Type")
        output_textbox = gr.outputs.Textbox(label="IWX Bot")
        output_textbox.show_copy_button = True
        output_textbox.lines = 10
        output_textbox.max_lines = 10

        if record_feedback:
            interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=output_textbox,
                                     theme="nota-ai/theme",
                                     title="IWX CHATBOT", allow_flagging="manual", flagging_callback=MysqlLogger(),
                                     flagging_options=data)
        else:
            interface = gr.Interface(fn=chatbot, inputs=[choice, msg], outputs=output_textbox,
                                     theme="nota-ai/theme",
                                     title="IWX CHATBOT", allow_flagging="never")
        if use_queue:
            interface.queue().launch(debug=debug, share=share_ui)
        else:
            interface.launch(debug=debug, share=share_ui)
