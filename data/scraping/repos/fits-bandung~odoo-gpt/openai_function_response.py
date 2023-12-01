import xmlrpc.client
import datetime
import os
import json
 
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, PromptTemplate, LLMChain
from langchain.agents import OpenAIFunctionsAgent, initialize_agent, Tool, AgentExecutor, AgentType, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory, ConversationBufferWindowMemory,ConversationEntityMemory
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser, RetryWithErrorOutputParser


from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

import sys
sys.path.append('j:\My Drive\FPP PROJECT\IT - MULTI MEDIA\REPO\odoo-gpt\odoo-gpt')
from utils.database import init_app, db_sqlalchemy, app
from utils.database import User as User
from utils.database import Message as Message
from utils.database import reset_memory, call_memory


from utils.get_credential import get_credentials, is_valid_token
from pydantic import BaseModel, Field
from typing import List, Optional

import jsonpickle

import streamlit as st


 


class OdooData(BaseModel):
    model: str
    fields: List[str]
    filter: List[List[str]]
    order: str
    limit: int


class OdooBrowseDataSchema(BaseModel):
    odoo_data: List[OdooData]


def OdooBrowse(data: OdooBrowseDataSchema):

    credentials_data = get_credentials(
        '47baa3066fd400e564980ff8acacca67d7436559cc1a1c49600a36364de03ac2')
    # print(f'Credentials : {credentials_data}')

    url = credentials_data['url']
    db = credentials_data['db']
    username = credentials_data['username']
    password = credentials_data['password']
    phone = credentials_data['phone_number']

    # chek uid from odoo
    common = xmlrpc.client.ServerProxy('{}/xmlrpc/2/common'.format(url))
    models = xmlrpc.client.ServerProxy('{}/xmlrpc/2/object'.format(url))

    common = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/common'.format(db))
    uid = common.authenticate(db, username, password, {})

    odoo_data = data[0]

    model = odoo_data['model']
    fields = odoo_data['fields']
    filter_list = odoo_data['filter']
    filter = [tuple(sublist) for sublist in filter_list]

    order = odoo_data['order']
    limit = odoo_data['limit']

    params = {'fields': fields}
    if limit is not None:
        params['limit'] = limit
    if order is not None:
        params['order'] = order

    # print(f'\n\nParameters : {params}')
    # print(f'Filter : {filter}')

    run_str = f"models.execute_kw('{str(db)}', '{str(uid)}', '{str(password)}', '{str(model)}', 'search_read', [{str(filter)}], {str(params)})"
    # print(f'Command : {run_str}')

    try:
        data = eval(run_str)
        # print(f'Data : {data}')

    # data = models.execute_kw(db, uid, password, model, 'search_read', filter, params)
    except Exception as e:
        print(e)
        error_str = '{"error":' + str(e) + '}'
        # Handle the error here
        data = json.loads(error_str)

    return data




def handle_error(e):
    print(f"Error: {e}")
    return f"Error: {e}"


def prepare_message(phone, incoming_message=None):

    OPENAI_API_KEY = os.environ['OPENAI_KEY']
    SERPAPI_API_KEY = os.environ['SERPAPI_API_KEY']
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                     openai_api_key=OPENAI_API_KEY)

    search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    db = SQLDatabase.from_uri("sqlite:///user_data.db")
    
    
    template = """This is a conversation between a human and a bot:

    {chat_history}

    Write a summary of the conversation for {input}:
    """

    

    prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)

    memory_key = "chat_history" 
    memory = ConversationBufferMemory(memory_key=memory_key)
    # memory = ConversationEntityMemory(llm=llm, memory_key=memory_key)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    summry_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
    )


    search = SerpAPIWrapper()


    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Gunakan ketika Anda perlu menjawab pertanyaan tentang peristiwa terkini. Jawab dengan bahasa sesuai dengan pertanyaan, atau dalam bahasa indonesia secara default",
            verbose=True,
            
            ),
        Tool(
            name="Summary",
            func=summry_chain.run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
        ),
        ]

    today = datetime.datetime.now().strftime("%d %B %Y")
    tool = Tool(
        name="OdooBrowse",
        func=OdooBrowse,
        args_schema=OdooBrowseDataSchema,
        description=f"""Mencari data dari sistem ERP Odoo versi 12 dengan orm odoo. berikan output sesuai dengan 'args_schema'.
            Saat melakukan filter gunakan sebisa mungkin operator 'ilike' alih-alih '='. Sebagai model kembalikan model sebagai model odoo versi 12. Jangan gunakan tools ini selain untuk mencari data seputar bisnis, dan yang terkait dengan HRD, Marketing/Sales, Keungan, Project, Operation.
            Hari ini adalah tanggal {today}""",
        verbose=True,
         
    )
    tools.append(tool)

    prefix = """Lakukan percakapan dengan manusia, jawab pertanyaan berikut sebaik mungkin. Anda memiliki akses ke Tools berikut:"""
    suffix = """Mulai!"

    {chat_history}
    Pertanyaan: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )


    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True, 
    )






    # from utils.get_credential import get_credentials, is_valid_token





    # K=50 # We set a low k=2, to only keep the last 2 interactions in memory
    # system_message = SystemMessage(
    #     content="Bantu sebaik mungkin untuk bisa menjawab dengan mengunakan Tools yang tersedia. Apabila ada pertanyaan yang sulit dijawab, jawab secara positif seperti halnya manusia menjawab."  
    # )

    # prompt = OpenAIFunctionsAgent.create_prompt(
    #     system_message=system_message,
    #     extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]  
    # )

    # prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE

    #Reset Entity Memory ketika pesan masuk adalah "reset"
    if incoming_message is not None and incoming_message.lower() == "reset":
        print('\n\nMelakukan reset memori...\n\n')
        buf_memory_json = None
        incoming_message = reset_memory(phone)

    # # memory = ConversationBufferMemory(k=K, memory_key=memory_key)

    buf_memory_json = call_memory(phone)
    if buf_memory_json is None:
        K=50 # We set a low k=2, to only keep the last 2 interactions in memory
        memory = ConversationEntityMemory(llm=llm, k=K, memory_key=memory_key)
        # memory = ConversationEntityMemory(llm=llm, k=K, memory_key=memory_key)
    else:
        memory = jsonpickle.decode(buf_memory_json)


    # memory = ConversationBufferMemory(k=K)
    print(f'\n\nMemory Sebelum Predict: {memory}\n\n')

    while True:

        incoming_message = input("You: ")
        # response = openai_agent.run(incoming_message)
       
        try:
            response = agent_chain.run(incoming_message)
           
       
        except Exception as e:
            
            print(f'\n\nError: {str(e)}\n\n')
            response = f'Error: {str(e)}'  




        #Menyimpan Entity Memory ke database
        buf_memory = agent_chain.memory
        # buf_memory = openai_agent.memory
        
        print(f'\n\nResponse : {response}\n')
        print(f'Memory : {buf_memory}')

        with app.app_context():
            user_query = User.query.filter_by(phone_number=phone).first()
            if user_query is None:
                print(f"No user found with phone: {phone}")
            else:
                print(f"User found with phone: {phone}")
                try:
                    buf_memory_jsonpickle = jsonpickle.encode(buf_memory)
                    user_query.entity_memory = buf_memory_jsonpickle
                    db_sqlalchemy.session.commit()
                    print(f"\n\nSuccessfully updated Memory (field entity_memory) for User with phone: {phone}")
                except Exception as e:
                    print(f"\n\nFailed to update entity memory: {str(e)}")
    
    return response



prepare_message('628112227980')
