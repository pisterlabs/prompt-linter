from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus

from lib.config import config
from lib.environment_variables import load_environment_variables

load_environment_variables()


def enriched_prompt(prompt, response_format="markdown"):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = Milvus(embedding_function=embeddings)
    retriever = db.as_retriever()
    cfg = config()
    model = ChatOpenAI(model=cfg["model"])  # 'ada' 'gpt-3.5-turbo' 'gpt-4',
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    result = qa({"question": f"Answer the following question in {response_format}: {prompt}", "chat_history": []})
    return result
