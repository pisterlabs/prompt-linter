from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from pinecone import Index, list_indexes

llm = ChatOpenAI()

template = """
You are Helpster, a chatbot that helps users with their queries about AICSSYC.
Answer the question in your own words as truthfully as possible from the context given to you.
If you do not know the answer to the question, respond with "I don't know. Can you ask another question".
If questions are asked where there is no relevant context available, respond with "Please ask a question related to AICSSYC"

Context: {context}

{chat_history}

Human: {question}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"], template=template
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, k=3
)

embedding = OpenAIEmbeddings()

index = Index("helpster")

vectordb = Pinecone(
    index=index,
    embedding=embedding,
    text_key="data",
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
)


def get_reply(msg):
    return chain.run(msg)
