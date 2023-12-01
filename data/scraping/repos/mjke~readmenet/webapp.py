#!/usr/bin/env python

# 2023/05 mjke

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# conversation memory
# https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


import gradio as gr


HAS_MEMORY = False

OPEN_AI_MODEL_NAME = "gpt-3.5-turbo" # or "gpt-4"

DIR_DATA = '/app/data'
DIR_DATA_DOCS = f'{DIR_DATA}/docs'
DIR_CHROMA_DB = f'{DIR_DATA}/chroma'

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 0

_LANGCHAIN_COLLECTION = 'langchain'
SEARCH_K = 3 # for gpt-3.5-turbo

CHAIN_TYPE = 'stuff' # https://docs.langchain.com/docs/components/chains/index_related_chains
VERBOSE = True


from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
# TODO possibly better
# from langchain.prompts.prompt import PromptTemplate
# # https://github.com/hwchase17/langchain/blob/master/langchain/chains/conversational_retrieval/prompts.py
# _CQ_TEMPLATE = """The following is a friendly conversation between a human and an AI Assistant. Given the conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
# ----
# Conversation History:
# {chat_history}
# ----
# Follow Up Question:
# {question}
# ----
# Standalone Question:"""
# CUSTOM_CONDENSE_PROMPT = PromptTemplate.from_template(template=_CQ_TEMPLATE)


qa_chain = None # global variable


def get_chromadb() -> Chroma:
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=DIR_CHROMA_DB)


def get_vectorstore_sources() -> list:
    docs = get_chromadb().get(include=["metadatas"])
    return list(set([s['source'] for s in docs['metadatas']]))


def del_vectorstore_docs(sources:str):
    if len(sources) > 0:
        db = get_chromadb()
        collection = db._client.get_collection(
            name=_LANGCHAIN_COLLECTION,
            embedding_function=OpenAIEmbeddings())
        for s in sources:
            collection.delete(where={ "source" : s })
        db.persist()
    return gr.update(choices = get_vectorstore_sources())


def safe_load_vectorstore(docs_raw:list) -> tuple:
    # check for docs already loaded (via metadata source in chromadb collection)
    existing = get_vectorstore_sources()
    docs = [d for d in docs_raw if d.metadata['source'] not in existing]

    # chunk docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_documents(docs)

    if len(docs) > 0:
        # calculate embeddings ($$)
        db = Chroma.from_documents(
            texts,
            embedding=OpenAIEmbeddings(), # note! uses `embedding` (not _function)
            persist_directory=DIR_CHROMA_DB)
        db.persist()
    return len(docs), len(docs) + len(existing)


def load_chain(db = None) -> None:
    if db is None: 
        db = get_chromadb()

    llm = ChatOpenAI(
        temperature=0,
        model_name=OPEN_AI_MODEL_NAME)
    
    global qa_chain
    if HAS_MEMORY:
        question_generator = LLMChain(
            llm=llm,
            #prompt=CUSTOM_CONDENSE_PROMPT,
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=VERBOSE)

        doc_chain = load_qa_with_sources_chain(
            llm,
            chain_type=CHAIN_TYPE,
            verbose=VERBOSE)

        qa_chain = ConversationalRetrievalChain(
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            retriever=db.as_retriever(search_kwargs={"k": SEARCH_K}),
            verbose=VERBOSE)
        
    else:
        # this works pretty well (but no memory)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=CHAIN_TYPE,
            retriever=db.as_retriever(),
            verbose=VERBOSE)
    

def load_data_directory(data_path:str = DIR_DATA) -> tuple:
    loader = DirectoryLoader(data_path, glob="*", use_multithreading=True)
    n_newdocs, n_sources = safe_load_vectorstore(loader.load())
    return \
        f"Loaded {n_newdocs} new. vectorDB contains {n_sources} total.", \
        gr.update(choices = get_vectorstore_sources())


def run_query(query:str, chat_history) -> tuple:
    if qa_chain is None:
        load_chain()
    if HAS_MEMORY:
        history = [(q, a) for q, a in chat_history]
        response = qa_chain({'question': query, 'chat_history': history})
        reply = response['answer']
    else:
        response = qa_chain(query)
        reply = response['result']
    chat_history.append((query, reply))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# READMENET")
    with gr.Tab("Chat"):
        gr.Markdown(f"Chat memory is{' **NOT**' if not HAS_MEMORY else ''} active")
        chatbot = gr.Chatbot()
        text_query = gr.Textbox(label="Query", lines=6, placeholder="Query here...")
        button_clear = gr.Button("Clear")
    with gr.Tab("Load docs"):
        data_dir = gr.Textbox(label="Data directory", value=DIR_DATA_DOCS)
        text_outcome = gr.Textbox(label="Outcome")
        button_load = gr.Button("Load docs into vectorDB")
    with gr.Tab('Manage docs'):
        boxes_sources = gr.CheckboxGroup(get_vectorstore_sources(), label="Docs", info="Stored in vectordb")
        button_delete = gr.Button("Delete selected docs")

    # chatbot
    text_query.submit(run_query, inputs = [text_query, chatbot], outputs = [text_query, chatbot])
    button_clear.click(lambda: None, None, chatbot, queue = False)

    # config
    button_load.click(load_data_directory, inputs=data_dir, outputs=[text_outcome, boxes_sources])
    button_delete.click(del_vectorstore_docs, inputs=boxes_sources, outputs=boxes_sources)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")



# ============================ NOTES =====================

# ## XX TODO ATTEMPTS at manual templates
# ## Note as good as RetrievalQA nor `load_qa_with_sources_chain` though

# # from langchain.chains.question_answering import load_qa_chain

# # https://github.com/hwchase17/langchain/blob/master/langchain/chains/question_answering/stuff_prompt.py
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# _SYS_TEMPLATE = """You are a constructively critical scientific Reviewer who gives feedback on NIH grant proposals and journal papers. You think your logic through step by step and explain your reasoning in succinct and thoughtful language.

# Use the following pieces of context to answer the users question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ----------------
# {context}"""
# messages = [
#     SystemMessagePromptTemplate.from_template(_SYS_TEMPLATE),
#     HumanMessagePromptTemplate.from_template("{question}"),
# ]
# CUSTOM_CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


# # # based on `from langchain.chains.conversational_retrieval.prompts import QA_PROMPT`
# # # https://github.com/hwchase17/langchain/blob/master/langchain/chains/retrieval_qa/prompt.py
# # _QA_TEMPLATE = """You are a constructively critical scientific Reviewer who gives feedback on NIH grant proposals and journal papers. You think your logic through step by step and explain your reasoning in succinct and understandable language. You always double-check your work. 

# # Use the following pieces of context to answer the question at the end. If you don't know the answer, truthfully say you do not know.
# # ----
# # Context:
# # {context}
# # ----
# # Question:
# # {question}
# # ----
# # Helpful Answer:"""
# # CUSTOM_QA_PROMPT = PromptTemplate.from_template(template=_QA_TEMPLATE)


# TODO
# # more memory stuff per https://github.com/hwchase17/langchain/issues/2303#issuecomment-1548837756
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
# # https://github.com/hwchase17/langchain/issues/2303
# # https://github.com/hwchase17/langchain/issues/2303#issuecomment-1536114140
# # https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html#conversationalretrievalchain-with-map-reduce

# # works ok (but not better than just handling chat history anyhow in gradio) - window version might be interesting
# memory = ConversationSummaryBufferMemory(
#     llm=llm,
#     output_key='answer',
#     memory_key='chat_history',
#     return_messages=True)
# (not working): VectorStoreRetrieverMemory



# TODO - stdout reroute doesn't work in containers
# import sys
# import logging

# PATH_LOG = f'{DIR_DATA}/chat.log'

# class StreamToLogger(object):
#     """
#     Fake file-like stream object that redirects writes to a logger instance.
#     src: https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
#     """
#     def __init__(self, logger, level):
#         self.logger = logger
#         self.level = level
#         self.linebuf = ''

#     def write(self, buf):
#         for line in buf.rstrip().splitlines():
#             self.logger.log(self.level, line.rstrip())

#     def flush(self):
#         pass

# logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#         filename=PATH_LOG,
#         filemode='a'
#         )
# log = logging.getLogger('chatlog')
# #sys.stdout = StreamToLogger(log, logging.INFO) # raises exception in container
# sys.stderr = StreamToLogger(log, logging.ERROR)