import openai
import sys
sys.path.append('../..')
from decouple import config

import panel as pn  # GUI
pn.extension()

openai.api_key  = config('OPENAI_API_KEY')

llm_name = "gpt-3.5-turbo"

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = '/Users/lxz/Desktop/openai/chroma_experiment/chatbot_db/x_db_copy'
embedding = OpenAIEmbeddings(openai_api_key = config('OPENAI_API_KEY'))
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(openai_api_key =config('OPENAI_API_KEY'),model_name=llm_name, temperature=0)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "你是谁？"
result = qa({"question": question})
print(result['answer'])

question = "为什么你称呼自己是这个人？有什么依据"
result = qa({"question": question})
print(result['answer'])
