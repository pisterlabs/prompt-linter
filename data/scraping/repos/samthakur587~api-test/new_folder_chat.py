from fastapi import FastAPI, Form, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from langchain import hub
from uuid import uuid4
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from fastapi import FastAPI,UploadFile, File
from fastapi.responses import JSONResponse
app = FastAPI()
import uvicorn
import requests
import os
from langchain.prompts import PromptTemplate
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import json
    
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
openai_api_key = os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

from langchain.tools import BaseTool, StructuredTool, Tool, tool


# Initialize FastAPI app
app = FastAPI()

# Create a dictionary to store user sessions
user_sessions = {}
chat_agent_mem={}
class InputData(BaseModel):
    name: str
    case_no: int
    background: str

class InputDataList(BaseModel):
    data: List[InputData]
def load(markdown_content: str,choice:int):
    headers_to_split_on = [
    ("#", "Case Number"),
    ("##", "Case No."),
    ("###", "Name of case"),
    ("####","Subheadings")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits1 = markdown_splitter.split_text(markdown_content)
    

    chunk_size = 2500
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(md_header_splits1)


    db = FAISS.from_documents(all_splits, embeddings)
    retriever = db.as_retriever()

    llm_premium = ChatOpenAI(temperature = 0.4, openai_api_key=openai_api_key,model_name="gpt-4",max_tokens=6000)
    llm_normal = ChatOpenAI(temperature = 0.4, openai_api_key=openai_api_key,max_tokens = 2050)
    if choice ==1:
        llm = llm_premium
    else:
        llm = llm_normal

    rag_prompt = hub.pull("rlm/rag-prompt")

    template = """Utilize the provided context to address the following question. If you encounter a situation where you don't possess the necessary information to provide a conclusive answer, kindly state your lack of knowledge and refrain from making assumptions. Please conclude with a courteous "thanks for asking!Compfox AI👋😊" at the end of your response.

Context: {context}
Question: {question}
Helpful Answer:
"""
    rag_prompt_custom = PromptTemplate.from_template(template)


    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
    return rag_chain




    memory_key="chat_history"
    
    llm = ChatOpenAI(temperature = 0.4, openai_api_key=openai_api_key)
    new_memory = ConversationSummaryBufferMemory(
            llm=llm,
            output_key='answer',
            memory_key=memory_key,
            return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
    memory=new_memory,#memory[user_id],
    chain_type="map_rerank",
    retriever= db.as_retriever(search_kwargs={"k": 3, "include_metadata": True}),
    return_source_documents=True,
    get_chat_history=lambda h : h,
    verbose=True)
    return qa

    @tool("search", return_direct=True)
    def search_document(query: str) -> str:
        """Searches and returns documents regarding the legal cases"""
        docs = retriever.get_relevant_documents(query)
        return str(docs)
    tools = [search_document]
    memory_key="demo_ne"


    memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

    system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Feel free to use any tools available to look up "
                "relevant information, only if neccessary"
                "Based on the tools resuly, answer nicely"
                "don't return json"
            )
    )
    prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
        )


    llm = ChatOpenAI(temperature = 0.4, openai_api_key=openai_api_key)

    # #using only chain
    # qa_chain = RetrievalQA.from_chain_type(
    # llm,
    # retriever=retriever.get_relevant_documents(query)
    # )


    #new method
    
    agent_executor = create_conversational_retrieval_agent(llm=llm, tools=tools, verbose=True)
    
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    
    
    agent_executor2 = AgentExecutor(agent=agent,memory=memory, tools=tools,verbose=True)
    

    return agent_executor

@app.post("/add-data/{user_id}/{session_id}")
async def add_data(user_id: str, session_id: str,choice:int, data: InputDataList):


    #choice ==1 means premium

    #Check for if allready existing 3 sessions are there.
    if user_id in user_sessions and len(user_sessions[user_id]) >= 3:
        return {"data": "Please delete the existing sessions to continue. You already have 3 existing sessions."}
    # Check if the user session exists, if not, create a new one
    if user_id not in user_sessions:
        user_sessions[user_id] = {}

    # Check if the session ID exists, if not, create a new one
    if session_id not in user_sessions[user_id]:
        user_sessions[user_id][session_id] = []

    # Process and store the data for the session
    num_to_string = {
        1: "First",
        2: "Second",
        3: "Third",
        4: "Fourth",
        5: "Fifth",
        6: "Sixth",
        7: "Seventh",
        8: "Eighth",
        9: "Ninth",
        10: "Tenth"
    }
    
    
    
    session_data = user_sessions[user_id][session_id]
    for i, input_data in enumerate(data.data):
        session_data.append(
            f"#case {input_data.case_no}\n"
            f"## {num_to_string[i + 1]} case\n"
            f"### {input_data.name}\n"
            f"#### Background:\n{input_data.background}"
        )

    markdown_content = "\n".join(user_sessions[user_id][session_id])
    chat_agent_mem['agent'] = load(markdown_content,choice)
    chat_agent_mem['user_id']=user_id
    chat_agent_mem['session_id'] = session_id
    print(data.data)
    return {"data": data.data}


@app.post("/chat/{user_id}/{session_id}")
async def process_text(user_id: str, session_id: str, input_text: str):
    # Check if the user session and session ID exist, and if not, create a new one
    if user_id not in user_sessions:
        user_sessions[user_id] = {}
    if session_id not in user_sessions[user_id]:
        user_sessions[user_id][session_id] = []

    if chat_agent_mem['session_id'] == session_id and chat_agent_mem['user_id']==user_id:
        agent_now = chat_agent_mem['agent']
        # agent_res = agent_now({"question":input_text})
        agent_res = agent_now.invoke(input_text)
        return {"result":agent_res,"status":"done"}

    # Load the markdown content from the session
    markdown_content = "\n".join(user_sessions[user_id][session_id])
    
    # Perform processing on the input text
    agent_executor2 = load(markdown_content)
    result = agent_executor2({"question": input_text})
    output = result['output']
    print(output)
    return {"output": output,"result":result,"status":"done"}
