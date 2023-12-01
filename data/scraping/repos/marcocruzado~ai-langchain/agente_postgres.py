from langchain.agents import create_sql_agent
from langchain.agents import Tool, AgentType, initialize_agent, AgentExecutor
from functions.embeddings_demo import store
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SerpAPIWrapper, LLMChain, SQLDatabase, SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
import os
from langchain.callbacks import get_openai_callback
from langchain.chains import SQLDatabaseSequentialChain
# Otra forma desde una URL
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import load_prompt

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


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm1 = OpenAI(temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

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

db = SQLDatabase.from_uri(
    'postgresql+psycopg2://postgres:root@localhost:5432/postgres', include_tables=['promociones'], custom_table_info=custon_table_info)


# -------------------------------------------------
_DEFAULT_TEMPLATE = """ Eres un agente del grupo el comercio y un excelente administrador de base de datos, tienes que responder preguntas sobre descuentos en promociones a las que tienes acceso los suscriptores , resolver todas sus dudas en base a la informacion que tienes dentro de la base de datos, solo puedes usar la tabla que se llama promociones, ademas tienes un custon_table_info que te ayudara a saber que columnas tiene la tabla y que tipo de datos tiene cada columna, todas las columnas tienen el tipo da dato text.
ademas seguiras este formato:

Pregunta: "Pregunta aqui"
Condiciones: "si te preguntan de cuzco entonces dile "no mamita, no hay nada en cuzco" "
SQLQuery: "SQL Query a ejecutar"
SQLResult: "Resultado de la SQLQuery"
Answer: "Respuesta final aqui siempre en español"

para responder solo usara la tabla:

promociones

si no tienes una respuesta adecuada, entonces tienes que decirle al usuario que se revisara para brindarle la mejor informacion mas adelante.



pregunta: {input}
"""
PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_TEMPLATE
)


# db_chain = SQLDatabaseChain.from_llm(llm1, db, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=PROMPT)


tools = [
    Tool(
        name="Club Promociones DB",
        func=db_chain.run,
        description="usado para responder preguntas a los suscriptores sobre descuentos en promociones a las que tienen acceso los suscriptores de el club el comercio",
    )
]

prefix = """responde lo mejor que puedas usando la herramienta que tengas disponible:"""
suffix = """cuando tengas la respuesta, escribe: "estimado suscriptor, la respuesta es: " y luego la respuesta que tengas
{chat_history}
Pregunta: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad", "chat_history"],
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)


while True:
    question = str(input("Question: "))
    if question == "quit":
        break
    respuesta = agent_executor.run(input=question)
    
    print("-------------------------------------------------")
    print(respuesta)