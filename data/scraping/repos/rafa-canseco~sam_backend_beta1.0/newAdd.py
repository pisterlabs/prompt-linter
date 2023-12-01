from decouple import config
from langchain.llms import OpenAI
import openai
import os
from langchain.callbacks import get_openai_callback
#para zappier
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
# para googlesearch
from langchain.agents import tools
from langchain.agents import load_tools
from langchain.agents import initialize_agent,Tool
#para wikipedia
from langchain.agents import AgentType,load_tools
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import  DuckDuckGoSearchRun ,BaseTool

#Para resumen de youtube
from langchain.document_loaders import  YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ssl

#Estructurar data desordenada
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.llms import OpenAI
import pandas as pd
import json

#leer un pdf y conversar con el
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#resumir un url
from langchain import OpenAI
from langchain.document_loaders import WebBaseLoader,YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from decouple import config


#enviroment variables
os.environ["OPENAI_API_KEY"] =config("OPEN_AI_KEY")
openai.api_key = config("OPEN_AI_KEY")
ZAPIER_NLA_API_KEY=config("ZAPIER_NLA_API_KEY")
SERPAPI_API_KEY =config("SERPAPI_API_KEY")
WOLFRAM_ALPHA_APPID= config("WOLFRAM_ALPHA_APPID")
wikipedia = WikipediaAPIWrapper()


#Codigo para usar zapier y dar instrucciones sobre otras apps
def instruction(message_decoded):
    with get_openai_callback() as cb:
        llm=OpenAI(temperature=0,openai_api_key=openai.api_key)
        zapier = ZapierNLAWrapper(zapier_nla_api_key=ZAPIER_NLA_API_KEY)
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        agent = initialize_agent(toolkit.get_tools(),llm,agent="zero-shot-react-description",verbose =True)
        agent.run(message_decoded)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")


#Codigo para usar Google search y wolfram alpha, y wikipedia
def search(message_decoded):
    with get_openai_callback() as cb:
        ssl._create_default_https_context = ssl._create_stdlib_context
        llm = OpenAI(temperature=0,openai_api_key=openai.api_key)
        tool_names = ["serpapi","wolfram-alpha","wikipedia","llm-math"]
        tools = load_tools(tool_names,serpapi_api_key=SERPAPI_API_KEY,wolfram_alpha_appid=WOLFRAM_ALPHA_APPID,wikipedia= wikipedia,llm=llm)
        search_duck = DuckDuckGoSearchRun()
        duckduckgotool_duck = [Tool(
                name='DuckDuckGo Search',
                func =search_duck.run,
                description="useful for when you need to do a search on the internet to find information that another tool cant find.Be specific with your input"
        )]
        tools.extend(duckduckgotool_duck)
        agent = initialize_agent(tools, llm ,agent="zero-shot-react-description",verbose=True)
        response = agent.run(message_decoded)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return response

# resumir una liga de youtube
def youtube_resume(url):
    with get_openai_callback() as cb:
        ssl._create_default_https_context = ssl._create_stdlib_context
        loader = YoutubeLoader.from_youtube_url(url)
        result = loader.load()
        llm=OpenAI(temperature=0,openai_api_key=openai.api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        chain = load_summarize_chain(llm,chain_type="map_reduce",verbose=False)
        resume = chain.run(texts)
        print("---------------------------")
        print("Resumen:   ")
        print(resume)
        print("---------------------------")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return resume
    

    
# resumir una liga de youtube
def url_resume(url):
    with get_openai_callback() as cb:
        loader = WebBaseLoader(url)
        result = loader.load()
        llm=OpenAI(temperature=0,openai_api_key=openai.api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
        texts = text_splitter.split_documents(result)
        chain = load_summarize_chain(llm,chain_type="map_reduce",verbose=False)
        resume = chain.run(texts)
        print("---------------------------")
        print("Resumen:   ")
        print(resume)
        print("---------------------------")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return resume
    
def preguntar_url(url,question):
    with get_openai_callback() as cb:
        loader = WebBaseLoader(url)
        result = loader.load()
        llm=OpenAI(temperature=0,openai_api_key=openai.api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
        texts = text_splitter.split_documents(result)
         #Seleccionar los embedings
        embeddings = OpenAIEmbeddings()
        #crear un vectorstore para usarlo de indice
        db=Chroma.from_documents(texts,embeddings)
        #revela el index en una interfaz a regresar
        retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":2})
        #crea una cadena para responder mensajes
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),chain_type="stuff",retriever=retriever,return_source_documents=True)
        query=question
        result= qa({"query":query})
        print(result)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result

def preguntar_youtube(url,question):
    with get_openai_callback() as cb:
        ssl._create_default_https_context = ssl._create_stdlib_context
        loader = YoutubeLoader.from_youtube_url(url)
        result = loader.load()
        llm=OpenAI(temperature=0,openai_api_key=openai.api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
        texts = text_splitter.split_documents(result)
         #Seleccionar los embedings
        embeddings = OpenAIEmbeddings()
        #crear un vectorstore para usarlo de indice
        db=Chroma.from_documents(texts,embeddings)
        #revela el index en una interfaz a regresar
        retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":2})
        #crea una cadena para responder mensajes
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),chain_type="stuff",retriever=retriever,return_source_documents=True)
        query=question
        result= qa({"query":query})
        print(result)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result



# leer un pdf
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# # #subir el documento
def pdf_pages(name_pdf,question):
    with get_openai_callback() as cb:
        
        loader = PyPDFLoader(name_pdf)
        documents = loader.load()
        # split el documento en pedazos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        #Seleccionar los embedings
        embeddings = OpenAIEmbeddings()
        #crear un vectorstore para usarlo de indice
        db=Chroma.from_documents(texts,embeddings)
        #revela el index en una interfaz a regresar
        retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":2})
        #crea una cadena para responder mensajes
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),chain_type="stuff",retriever=retriever,return_source_documents=True)
        query=question
        result= qa({"query":query})
        print(result)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result


#----------------------
#Limpiar Data    
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
import json

def dirty_data(categorias,data):
    with get_openai_callback() as cb:

            standarized_result = categorias
            user_input = data
            chat_model = ChatOpenAI(temperature=0,openai_api_key=openai.api_key, max_tokens=1000)
            #como queremos la respuesta estructurada, prompt
            response_schemas = [
                ResponseSchema(name="user_input", description="este el input del usuario"),
                ResponseSchema(name="standarized_result", description ="esta es la categoría estandarizada"),
                ResponseSchema(name="match_score", description="Una puntuación de 0 a 100 de lo cerca que cree que está la coincidencia entre la entrada del usuario y su coincidencia")
            ]
            #Cómo queremos ordenar la respuesta
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            #ver el promp template para formatearlo
            format_instructions = output_parser.get_format_instructions()
            #Crear el template 

            template ="""
            Se le dará varios tipos de categorías que estarán designadas en la variable Standarized Result.
            Encuentre la mejor coincidencia correspondiente a cada palabra del user_input con las  categorías existentes
            La coincidencia más cercana será la que tenga el significado semántico más cercano. No solo similitud de strings.

            {format_instructions}

            Envuelva su resultado final con corchetes abiertos y cerrados (una lista de objetos json)
            input_ INPUT:
            {user_input}

            STANDARIZED RESULT:
            {standarized_result}

            Tu Respuesta:

            """

            prompt = ChatPromptTemplate(
                messages=[
                HumanMessagePromptTemplate.from_template(template)
                ],
                input_variables=["user_input","standarized_result"],
                partial_variables={"format_instructions": format_instructions}
            )


            #User input
            _input =prompt.format_prompt(user_input=user_input,standarized_result=standarized_result)

            print(f"They are{len(_input.messages)} messages(s)")
            print(f"Type:{type(_input.messages[0])}")
            print("------------------")
            print(_input.messages[0].content)

            response = chat_model(_input.to_messages())
            print(type(response))
            print(response.content)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            return response

# #----- interactuar con data y extraer información

from kor import create_extraction_chain, Object, Text, Number
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


def abstraction(globalDescription, data, attributes, attributeValues, ejemplo,obj_global):
    with get_openai_callback() as cb:

        
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=2000,
            openai_api_key=openai.api_key
        )

        person_schema = Object(
            id=obj_global,
            description=globalDescription,
            attributes=[Text(id=attr['id'].replace(' ', '_').lower(), description=attr['description'])  for attr in attributes],
            examples=[(ejemplo,[{attr['id'].replace(' ', '_').lower(): attributeValues.get(attr['id'], {})} for attr in attributes]
                )
            ]
        )



        chain = create_extraction_chain(llm, person_schema, encoder_or_encoder_class='json')
        response = chain.predict_and_parse(text=data)['raw']
        print(response)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return response



# from kor import create_extraction_chain, Object, Text, Number
# import json
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI


# def abstraction(globalDescription, data, attributes, attributeValues, ejemplo,obj_global):
#     with get_openai_callback() as cb:
        
#         llm = ChatOpenAI(
#             model_name="gpt-3.5-turbo",
#             temperature=0,
#             max_tokens=2000,
#             openai_api_key=openai.api_key
#         )

#         person_schema = Object(
#             id=obj_global,
#             description=globalDescription,
#             attributes=[Text(id=attr['id'], description=attr['description']) if attr.get('type') == 'text' else Number(id=attr['id'], description=attr['description']) for attr in attributes],
#             examples=[(ejemplo,[{attr['id']: attributeValues.get(attr['id'], {})} for attr in attributes]
#                 )
#             ]
#         )



#         chain = create_extraction_chain(llm, person_schema, encoder_or_encoder_class='json')
#         response = chain.predict_and_parse(text=data)['raw']
#         print(response)
#         print(f"Total Tokens: {cb.total_tokens}")
#         print(f"Prompt Tokens: {cb.prompt_tokens}")
#         print(f"Completion Tokens: {cb.completion_tokens}")
#         print(f"Successful Requests: {cb.successful_requests}")
#         print(f"Total Cost (USD): ${cb.total_cost}")
#         return response

from kor import create_extraction_chain, Object, Text, Number
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import openai
from web3 import Web3,Account
from decimal import Decimal


openai.api_key = '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'
privada = '0x92bba9ec8422a919d7896fd954eef1797c887daf1fcb2bf9d97a2995a214987d'
address = '0xe73693D0e16e421270e3CB02Ea5f377b75E93e44'

def blockchain_tx(data):
    Alchemy = 'https://eth-sepolia.g.alchemy.com/v2/DfvwNaHxCJYvzlXbqL4Na6GdzTROAUoK'

    with get_openai_callback() as cb:
            
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=2000,
                openai_api_key=openai.api_key
            )
        
            person_schema = Object(
        
                id="transaction",
                description="Información de una transacción de crypto de una persona a otra persona ",
                
                # Notice I put multiple fields to pull out different attributes
                attributes=[
                    Text(
                        id="crypto",
                        description="nombre de la criptomoneda."
                    ),
                    Text(
                        id="qty",
                        description="cantidad de crypto que se va a mandar "
                    ),
                    Text(
                        id="nombre_receptor",
                        description="persona que va a recibir la crypto"
                    )
                ],
                examples=[
                    (
                        "Enviar 10 ether a orlando",
                        [
                            {"crypto": "ether"},
                            {"qty": "10"},
                            {"nombre_receptor" : "orlando"}
                        ],
                    )
                ]
            )
            text=data 

            chain = create_extraction_chain(llm, person_schema,encoder_or_encoder_class='json')
            response = chain.predict_and_parse(text=text)['data']
            print(response)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            #  Extracción de los valores del diccionario
            crypto_name = response['transaction'][0]['crypto']
            qty = response['transaction'][1]['qty']
            nombre_receptor = response['transaction'][2]['nombre_receptor']
            # Utilización de los valores extraídos
            print(f"Enviar {qty} {crypto_name} a {nombre_receptor}")

            if crypto_name == "link":
                token_address = '0x779877A7B0D9E8603169DdbD7836e478b4624789'
                if nombre_receptor == 'rafa':
                    destination = '0xe1A73Ea88BDC41D98C1bC2f5325881320031F02B'
                elif nombre_receptor == 'orlando':
                    destination = '0x6D082B729Fa74206DD5454148616C278515De955'
                else:
                # En caso de que el nombre no sea rafa ni orlando, se puede mostrar un mensaje de error y salir del programa
                    print('Nombre de receptor inválido')


                # Conectar a la cadena Ethereum
                Alchemy = 'https://eth-sepolia.g.alchemy.com/v2/DfvwNaHxCJYvzlXbqL4Na6GdzTROAUoK'
                w3=Web3(Web3.HTTPProvider(Alchemy))

                # Dirección del contrato del token y ABI
                token_contract_address = token_address
                token_intro = w3.to_wei(qty,'wei')
                token_abi = [
                    {
                        "constant": False,
                        "inputs": [
                            {
                                "name": "_to",
                                "type": "address"
                            },
                            {
                                "name": "_value",
                                "type": "uint256"
                            }
                        ],
                        "name": "transfer",
                        "outputs": [
                            {
                                "name": "",
                                "type": "bool"
                            }
                        ],
                        "payable": False,
                        "stateMutability": "nonpayable",
                        "type": "function"
                    }
                ]

                # Crear instancia del contrato
                token_contract = w3.eth.contract(address=token_contract_address, abi=token_abi)

                # Dirección del receptor y cantidad de tokens a enviar
                recipient_address = destination
                token_amount = token_intro*10**18

                # Crear transacción para llamar a la función transfer
                tx = token_contract.functions.transfer(recipient_address, token_amount).build_transaction({
                    'from': address,
                    'gas': 100000,
                    'gasPrice': w3.to_wei('1', 'gwei'),
                    'nonce': w3.eth.get_transaction_count(address)
                })

                # Firmar la transacción con la clave privada del remitente
                signed_tx = w3.eth.account.sign_transaction(tx, private_key=privada)

                # Enviar la transacción
                tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

                # Esperar a que se confirme la transacción
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                readable =w3.to_hex(tx_hash)
                print(w3.to_hex(tx_hash))
                print("listo")
                
                return readable

                
            elif crypto_name =="ether":

                if nombre_receptor == 'rafa':
                    destination = '0xe1A73Ea88BDC41D98C1bC2f5325881320031F02B'
                elif nombre_receptor == 'orlando':
                    destination = '0x6D082B729Fa74206DD5454148616C278515De955'
                else:
                    # En caso de que el nombre no sea rafa ni orlando, se puede mostrar un mensaje de error y salir del programa
                    print('Nombre de receptor inválido')

                    
                w3=Web3(Web3.HTTPProvider(Alchemy))

                nonce =w3.eth.get_transaction_count(address)
                

                transaction ={
                    'to': destination,
                    'from': address,
                    'value': w3.to_wei(qty,'ether'),
                    'gas': 21000,
                    'gasPrice': w3.eth.gas_price,
                    'nonce': nonce
                    
                }
                signed_tx = w3.eth.account.sign_transaction(transaction,privada)
                txn_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = w3.to_hex(txn_hash)
                print(receipt)
                print("envio realizado")
                return receipt
            
from langchain.document_loaders import ToMarkdownLoader
import os
from firebase_admin import credentials
from firebase_admin import storage
import openai
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex,SimpleDirectoryReader
import re


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
            
def load_url(liga, user):
    api_key = os.getenv("api_key_2markdown")
    storage_client = storage.bucket("samai-b9f36.appspot.com")

    print(liga)
    urls = liga
    print(urls)


    for i, url in enumerate(urls, start=1):
        loader = ToMarkdownLoader(api_key=api_key, url=url)
        docs = loader.load()
        if docs:
            content = docs[0].page_content
            pattern = r"!\[\]\(data:image\/\w+;base64,[^)]+\)"
            filtered_content = re.sub(pattern, "", content)
            carpeta_destino = "news-free-links"
            nombre_archivo = f"archivo{i}.txt"
            ruta_destino = f"{user}/{carpeta_destino}/{nombre_archivo}"

            carpeta_ref = storage_client.blob(f"{user}/{carpeta_destino}")
            if not carpeta_ref.exists():
                carpeta_ref.upload_from_string("")

            archivo_ref = storage_client.blob(ruta_destino)
            archivo_ref.upload_from_string(filtered_content)
            print("proceso terminado")

    #segunda parte guardar y vectorizar

    prefix = f"{user}/news-free-links/"

    local_folder = "./news-free-links"
    os.makedirs(local_folder,exist_ok=True)

    blobs = storage_client.list_blobs(prefix=prefix)

    for blob in blobs:
        nombre_archivo =blob.name.split("/")[-1]

        archivo_local = os.path.join(local_folder,nombre_archivo)
        blob.download_to_filename(archivo_local)
        print("noticias descargadas")


    documents = SimpleDirectoryReader('news-free-links').load_data()
    print("documentos listos")
    index = GPTVectorStoreIndex.from_documents(documents)
    print(index)
    index.storage_context.persist()
    # Ruta local de la carpeta "storage"
    ruta_carpeta_local = "storage"

    # Ruta de la carpeta en Firebase Storage
    ruta_carpeta_destino = f"{user}/storage-free-links"

    # Verificar si la carpeta existe en Firebase Storage
    carpeta_ref = storage_client.blob(ruta_carpeta_destino)
    if not carpeta_ref.exists():
        # Crear la carpeta si no existe
        carpeta_ref.upload_from_string("")

    # Subir la carpeta y sus archivos a Firebase Storage
    for root, dirs, files in os.walk(ruta_carpeta_local):
        for file in files:
            ruta_archivo_local = os.path.join(root, file)
            ruta_archivo_destino = os.path.join(ruta_carpeta_destino, file)
            archivo_ref = storage_client.blob(ruta_archivo_destino)
            archivo_ref.upload_from_filename(ruta_archivo_local)
            print("documentos vectorizados")

import os
from llama_index import StorageContext, load_index_from_storage
from dotenv import load_dotenv
import openai
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage



def pregunta_url_resumen(user):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    storage_client = storage.bucket("samai-b9f36.appspot.com")

    user = user
    prefix = f"{user}/storage-free-links/"

    local_folder = "./storage"
    os.makedirs(local_folder, exist_ok=True)

    blobs = storage_client.list_blobs(prefix=prefix)

    for blob in blobs:
        nombre_archivo = blob.name.split("/")[-1]
        archivo_local = os.path.join(local_folder, nombre_archivo)
        blob.download_to_filename(archivo_local)

    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    x =  """Escribe un resumen usando Bullet Points.
        -Bullet point format:
            -Separate each bullet point with a new line.
            -Each bullet point should be concise.
        """
    response = query_engine.query(x)
    print(response)
    return response

def pregunta_url_abierta(user,question):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    storage_client = storage.bucket("samai-b9f36.appspot.com")

    user = user
    prefix = f"{user}/storage-free-links/"

    local_folder = "./storage"
    os.makedirs(local_folder, exist_ok=True)

    blobs = storage_client.list_blobs(prefix=prefix)

    for blob in blobs:
        nombre_archivo = blob.name.split("/")[-1]
        archivo_local = os.path.join(local_folder, nombre_archivo)
        blob.download_to_filename(archivo_local)

    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    x = question
    response = query_engine.query(x)
    print(response)
    return response


