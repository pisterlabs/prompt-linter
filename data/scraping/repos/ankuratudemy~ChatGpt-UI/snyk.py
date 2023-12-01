import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

app = Flask(__name__)
port = 4000
CORS(app)

embeddings = OpenAIEmbeddings()
vectordbChroma = Chroma(embedding_function=embeddings,
                        persist_directory="./snyk_chroma_db")


@app.route("/", methods=["POST"])
def chat_completion():
    data = request.json
    cwe = data["cwe"]
    snyk_code = data["snyk_code"]
    query = data["query"]
    # userid = data["userid"]

    # history = PostgresChatMessageHistory(
    #     connection_string="postgresql://postgres:admin@localhost:5432/session_db",
    #     session_id=userid
    # )

    try:
        loader = WebBaseLoader([f"https://cwe.mitre.org/data/definitions/{cwe}.html", f"https://security.snyk.io/vuln/{snyk_code}"])
        data = loader.load()
        
        
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
        all_splits = text_splitter.split_documents(data)
        
        
        print(all_splits)

        # docsearch = Chroma.from_documents(all_splits, embeddings)
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        retriever=vectorstore.as_retriever()
        qa_chain = load_qa_chain(OpenAI(temperature=0.7), chain_type="refine", verbose=True)
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, please think rationally and answer from your own knowledge base 

        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.7), 
                                        chain_type="stuff", 
                                        retriever=retriever, 
                                        chain_type_kwargs=chain_type_kwargs)
        # qa = RetrievalQA.from_chain_type(OpenAI(temperature=0.7), chain_type="stuff", retriever=vectorstore.as_retriever())
        response = qa.run(query)
        
        # response = chain.run(input_documents=data, question=query)
        
        print(response)
        
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix(
            "Could not parse LLM output: `").removesuffix("`")
    # history.add_ai_message(response)
    return jsonify({"botResponse": response})


@app.route("/history", methods=["POST"])
def get_history():
    data = request.json
    userid = data["userid"]

    history = PostgresChatMessageHistory(
        connection_string="postgresql://postgres:admin@localhost:5432/session_db",
        session_id=userid
    )

    history_data = history.messages

    pairs = [
        {
            "chatPrompt": history_data[i].content,
            "botMessage": history_data[i+1].content,
        }
        # assume the list always start with HumanMessage and then followed by AIMessage
        for i in range(0, len(history_data), 2)
    ]

    return jsonify({"history": pairs})


if __name__ == "__main__":
    app.run(port=port)
