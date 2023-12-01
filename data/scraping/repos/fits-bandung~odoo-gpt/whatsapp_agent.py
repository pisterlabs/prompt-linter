from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import ZeroShotAgent, AgentExecutor, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.chat_models import ChatOpenAI

from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
 

import jsonpickle

from utils.database import db_sqlalchemy, app
from utils.database import User as User, inspect_db, call_memory
# from utils.whatsapp import prepare_message
from utils.tools import get_date_time, answer_general_query



load_dotenv('.credentials/.env')
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

search = GoogleSearchAPIWrapper()
# search = SerpAPIWrapper()

# get_date_time = 

tools = [
    # Tool(
    #     name="General Query",
    #     func=answer_general_query,
    #     description="berguna untuk menjawab pertanyaan umum dari berbagai topik yang tidak memerlukan sumber data eksternal. Gunakan seluruh bagian input sebagai query untuk mendapatkan jawaban melalui tools ini.",
    #     # return_direct=True,

    # ),

    
    Tool(
        name="Search",
        func=search.run,
        description="berguna ketika Anda perlu menjawab pertanyaan tentang informasi terkini, dan perlu melakukan search melalui internet.",
   
    ),
    
    
    Tool(
        name="Get Date and Time",
        func=get_date_time,
        description="berguna ketika Anda perlu menjawab pertanyaan tentang tanggal, hari dan jam (waktu) saat ini",
    ),
]




MODEL = 'gpt-3.5-turbo'
API_O = os.environ['OPENAI_KEY']







def get_memory(phone):
    
    buf_memory_json = call_memory(phone)
            
    if buf_memory_json is None:
        # memory = ConversationBufferMemory(memory_key="chat_history")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    else:
        memory = jsonpickle.decode(buf_memory_json)
        
    # memory = ConversationBufferMemory(memory_key="chat_history")
       
    return memory


def save_memory(phone, memory):
    with app.app_context():
        user_query = User.query.filter_by(phone_number=phone).first()
        # user_query = jsonpickle.encode(memory)

        if user_query is not None:
            user_query.entity_memory =jsonpickle.encode(memory)
            db_sqlalchemy.session.commit()


    
    return memory
        


def predict_gpt(phone_number, incoming_message):
    output = "..."
    total_cost = 0.0

    memory = get_memory(phone_number)
    # print(f'\n\nMemory from database (before) : {memory}')
   

    MODEL = 'gpt-3.5-turbo'

    prefix = """Anda adalah FujiBoy, assisten virtual yang akan membantu menjawab semua pesan masuk sebagai melalui aplikasi whatsapp.
    Anda memiliki pengetahuan yang luas dalam berbagai bidang, dan bisa membantu mulai dari menjawab pertanyaan hingga diskusi mendalam. 
    Berikan jawaban dalam bahasa Indonesia. 
    Anda memiliki akses ke tools berikut:"""
    suffix = """Mulai!

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""


    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )


   
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0,
                openai_api_key=API_O,
                model_name=MODEL,
                verbose=True,
                max_tokens=400,
                )

    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        # tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, 
        handle_parsing_errors="Check your output, make resume, and make sure it conforms!",
        max_iterations=5, 
        early_stopping_method="generate", 
        prompt=prompt, 
        memory=memory
    )



    with get_openai_callback() as cb:

        try:
            # output = Conversation.run(input=incoming_message)

            # output = agent_chain.run(input=incoming_message)
            output = agent.run(incoming_message)
            
            print(f'\nOutput: {output}\n')

            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (IDR): IDR {cb.total_cost*15000}\n")
        
            memory = save_memory(phone_number, memory)
            # print(f'\n\nMemory saved to database (after): {memory}')

        except Exception as e:
            # output = prepare_message(phone_number, incoming_message)
            print(f'Error: {e} [predict_gpt] line 109]')

        
        total_cost = cb.total_cost*15000


    return output, total_cost
   