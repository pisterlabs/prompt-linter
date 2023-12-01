from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain 
from langchain.chains import SQLDatabaseSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
import os



from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(os.environ["OPENAI_API_KEY"])

#### FUENTES DE INFORMACION"#####################

# tiene que buscar en google y devolver el primer resultado todas las busquedas sera en Peru y en español
search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"], params={
                        "engine": "google", "google_domain": "google.com", "gl": "pe", "hl": "es-419"})

# -------------------------------------------------

# postgresql+psycopg2://pguser:password@localhost:5433/doc_search

custon_table_info = {
    "Track": """
    CREATE TABLE promociones (
        id SERIAL NOT NULL, 
        descuento TEXT, 
        descuento_valor TEXT,
        descuento_descripcion TEXT,
        url_establecimiento TEXT,
        nombre TEXT,
        facebook TEXT,
        categoria TEXT,
        descripcion TEXT,
        direccion TEXT,
        horario TEXT,
        vigencia TEXT,
        condicion TEXT,
        telefono TEXT,
        correo TEXT,
        web TEXT,
        terminos TEXT,
        CONSTRAINT promociones_pkey PRIMARY KEY (id)
)

/*
3 rows from promociones table:
id      descuento       descuento_valor descuento_descripcion   url_establecimiento     nombre  facebook        categoria       descripcion      direccion       horario vigencia        condicion       telefono        correo  web     terminos
1       Hasta45%Dscto.  45%     HASTA -45% EN CURSOS TECNOLÓGICOS       https://clubelcomercio.pe/educacion-tecky-brains-tecky-brains-13758      TECKY BRAINS    https://www.facebook.com/sharer/sharer.php?u=https://clubelcomercio.pe/educacion-tecky-brains-tecky-    EducacióTecky Brains, institución educativa online promueve niños creadores y emprendedores para el futuro a     NaN     NaN     Del 01 de junio al 31 de octubre 2023    Comunícate al WhatsApp e indica tu DNI para la matrícula.       (01)  978941451 informes@teckybrains.fun        http://www.teckybrains.fun/      Descuento exclusivo para suscriptores y/o beneficiarios de las ediciones impresas y digitales de los     
2       Hasta53%Dscto.  53%     HASTA -53% EN DETERMINADOS PROGRAMAS    https://clubelcomercio.pe/educacion-berlitz-berlitz-adultos-1359BERLITZ ADULTOS  https://www.facebook.com/sharer/sharer.php?u=https://clubelcomercio.pe/educacion-berlitz-berlitz-adu    Educación       El programa Adultos es una clase con un profesor de idiomas online en vivo, que te permite aprender      Beneficio Online.       NaN     Del 01 al 31 de julio 2023.      Solicítalo vía correo a: ingrid.reyes@berlitz.com.pe  indicando tu DNI. (01)  017076479 ingrid.reyes@berlitz.com.pe      #       Descuento exclusivo para suscriptores y/o beneficiarios de las ediciones impresas y digitales de los
3       Hasta60%Dscto.  60%     HASTA -60% EN PROGRAMA KIDS & TEENS     https://clubelcomercio.pe/educacion-berlitz-berlitz-kids-teens-13596     BERLITZ KIDS & TEENS    https://www.facebook.com/sharer/sharer.php?u=https://clubelcomercio.pe/educacion-berlitz-berlitz-kid    Educación        Berlitz Kids & Teens es un programa único, virtual e interactivo para aprender inglés en vivo, en el    Beneficio OnlineNaN      Del 01 al al 31 de julio 2023   Solicítalo vía correo a: ingrid.reyes@berlitz.com.pe indicando tu DNI.  (01)  017076479 ingrid.reyes@berlitz.com.pe      #       Descuento exclusivo para suscriptores y/o beneficiarios de las ediciones impresas y digitales de los     
*/
    """
}

input_db = SQLDatabase.from_uri('postgresql+psycopg2://postgres:root@localhost:5432/postgres',
                                include_tables=['promociones'], custom_table_info=custon_table_info)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


db_agent = SQLDatabaseSequentialChain.from_llm(llm,
                                     database=input_db,
                                     verbose=True,
                                     input_key="input",
                                     )

# -------------------------------------------------

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Search in Database",
        func=db_agent.run,
        description="cuando necesites saber sobre promociones de los establecimientos que el club el comercio,numeros de telefono, correos, etc",
    ),
]



agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    handle_parsing_errors=True,
)


while True:
    user_input = str(input("User:"))
    agent_output = agent.run(input=str(user_input))
    print("Agent:", agent_output)