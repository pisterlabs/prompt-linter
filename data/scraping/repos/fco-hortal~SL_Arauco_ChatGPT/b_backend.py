# from langchain.sql_database import SQLDatabase
# from langchain import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

import openai
import os
from dotenv import load_dotenv

# Cargamos BD con langchain
db = SQLDatabase.from_uri("sqlite:///ecommerce.db")

# Cargamos variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
db_chain.run("How many employees are there?")

# # llm = OpenAI(temperature=0, verbose=True)
# # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# #pip install --upgrade openai

# # Creamos LLM
# llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# # Creamos cadena
# cadena = SQLDatabaseChain.from_llm(llm = llm, database = db, verbose=False)

# # Respuesta personalizada
# formato = """
# Data una pregunta del usuario:
# 1. crea una consulta de sqlite3
# 2. revisa los resultados
# 3. devuelve el dato
# 4. si tienes que hacer alguna aclaración o devolver cualquier texto que sea siempre en español
# #{question}
# """

# # Función de consultas

# def consulta(input_usuario):
#     consulta = formato.format(question = input_usuario)
#     resultado = cadena.run(consulta)
#     return(resultado)

# https://github.com/openai/openai-python.git
# pip install -e .
