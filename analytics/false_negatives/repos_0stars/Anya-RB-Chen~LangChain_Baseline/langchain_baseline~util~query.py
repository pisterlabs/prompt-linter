from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory

from util.config import Config
from util.database import Ringley_Database
import util.templates as templates

config = Config()
db_dir = config.get_db_dir()
vector_dir = config.get_vector_dir()
openai_key = config.get_openai_key()
config.connect_OPENAI()

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(templates.CONDENSE_QUESTION_PROMPT)
QA_PROMPT = PromptTemplate(template=templates.QA_PROMPT_FAISS, input_variables=[
                           "question", "context"])

def run_database(update_data=False):
    db = Ringley_Database()
    print("[INFO] Start updating data...")
    print("[ATTENTION] Please make sure you have the latest data in the data/update folder.")
    print("[ATTENTION] Do not modify the data in the data/update folder until the embedding is finished.")
    db.run_data_sync_faiss(update=update_data)

def load_retriever(vector_path=vector_dir, update_data=False):
    config.connect_OPENAI()
    embeddings = OpenAIEmbeddings()
    if update_data:
        run_database(update_data=update_data)
    vector = FAISS.load_local(vector_path, embeddings=embeddings)
    retriever = VectorStoreRetriever(vectorstore=vector)
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

def get_condense_prompt_qa_chain_update_data():
    model_name = config.get_model_name()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = load_retriever(update_data=True)
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
    "condense_prompt_update_data": get_condense_prompt_qa_chain_update_data,
}