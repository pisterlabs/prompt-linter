from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle
import os

from util.config import Config
import util.templates as templates

config = Config()
db_dir = config.get_db_dir()
config.connect_OPENAI()

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(templates.CONDENSE_QUESTION_PROMPT)
QA_PROMPT = PromptTemplate(template=templates.QA_PROMPT, input_variables=[
                           "question", "context"])


def load_retriever_pkl(db_dir=db_dir):
    # only for testing purposes
    with open(os.path.join(db_dir, "MASTER_FAQ_EXCEL.pkl"), "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever

def load_retriever(db_dir=db_dir):
    # Load all the embeddings from the db_dir
    # @TODO: the master faq is only for testing purposes
    with open(os.path.join(db_dir, "MASTER_FAQ_EXCEL.pkl"), "rb") as f:
        vectorstore = pickle.load(f)

    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_basic_qa_chain():
    model_name = config.get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory)
    return model


def get_custom_prompt_qa_chain():
    model_name = config.get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_condense_prompt_qa_chain():
    model_name = config.get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def condense_prompt_qa_chain_pkl():
    model_name = config.get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = load_retriever_pkl()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model

def get_qa_with_sources_chain():
    model_name = config.get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question['question'], "chat_history": history}
        result = model(new_input)
        history.append((question['question'], result['answer']))
        return result

    return model_func


chain_options = {
    "basic": get_basic_qa_chain,
    "with_sources": get_qa_with_sources_chain,
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain,
    "condense_prompt_pkl": condense_prompt_qa_chain_pkl
}