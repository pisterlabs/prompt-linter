from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationKGMemory,ConversationBufferMemory
from langchain.vectorstores import Chroma
from .ingest import *
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import os
def load_llm(PERSIST_DIRECTORY, SOURCE_DIRECTORY):
        os.environ["OPENAI_API_KEY"] = "YOUR API HERE"

        embeddings = OpenAIEmbeddings()
        
        CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
        )
        
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever()
        

        template = """
        You are a helpful AI assistant. you are given file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        Use as much detail as possible when responding.
        This is the following chat history available to you, refer to it and do not mention this usage in the answer {chat_history}

        context: {context}
        =========
        question: {question}
        ======
        """
        prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"], template=template)
        llm = ChatOpenAI(temperature=0.1,model_name='gpt-3.5-turbo')
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,memory=memory, return_source_documents=True,combine_docs_chain_kwargs={"prompt": prompt})
        
        return qa





def proc(query,PERSIST_DIRECTORY, SOURCE_DIRECTORY):
        #main_ingest()
        qa = load_llm(PERSIST_DIRECTORY, SOURCE_DIRECTORY)    
        res = qa({"question":query})
        answer,source = res['answer'] , res['source_documents']
        stranswer = str(answer)
        strsource = str(source)
        return stranswer
