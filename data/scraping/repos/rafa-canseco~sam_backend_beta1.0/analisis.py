import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from llama_index import GPTVectorStoreIndex,SimpleDirectoryReader,StorageContext, load_index_from_storage

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate)

from langchain.schema import (
    AIMessage,HumanMessage,SystemMessage
)
from langchain.callbacks import get_openai_callback
import json
import openai
from decouple import config
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import shutil


os.environ["OPENAI_API_KEY"] =config("OPEN_AI_KEY")
openai.api_key = config("OPEN_AI_KEY")


# Resumen sencillo
def resumen_sencillo(user):
    with get_openai_callback() as cb:
        storage_client = storage.bucket("samai-b9f36.appspot.com")
        folder_name = f"{user}/{user}_workflow"
        file_name = f"{folder_name}/test_data_{user}.txt"

        # Crear el directorio de destino si no existe
        os.makedirs(folder_name, exist_ok=True)

        blob = storage_client.blob(file_name)
        archivo_destino = f"./{file_name}"
        blob.download_to_filename(archivo_destino)

        with open(archivo_destino, "r") as f:
            contenido = f.read()
            print(contenido)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
        texts = text_splitter.create_documents([contenido])
        llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
        output = chain.run(texts)
        print(output)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return output



#Resumen con instrucciones precisas
def resumen_datos_personales(user):
    with get_openai_callback() as cb:
        storage_client = storage.bucket("samai-b9f36.appspot.com")
        folder_name = f"{user}/{user}_workflow"
        file_name = f"{folder_name}/test_data_{user}.txt"

        # Crear el directorio de destino si no existe
        os.makedirs(folder_name, exist_ok=True)

        blob = storage_client.blob(file_name)
        archivo_destino = f"./{file_name}"
        blob.download_to_filename(archivo_destino)

        with open(archivo_destino, "r") as f:
            contenido = f.read()
            print(contenido)


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=250)
        texts = text_splitter.create_documents([contenido])

        llm = ChatOpenAI(temperature=0,openai_api_key=openai.api_key)

        template = """
        Eres un análista de conversaciones para una empresa de marketing.
        Tu Compañía es lo que en la conversación viene como "role":"asssistant" y el usuario como "role":"user"
        Vas a analizar lo que el "role":"user" comenta.
        Tu meta es escribir un resumen de los comentarios del usuario, analizar sus sentimientos, y abstraer la información personal que él otorgue.
        Es importante que resaltes los puntos mas importantes de la conversación, si encuentras datos del usuario,entregalos en un formato JSON.
        No respondas con nada fuera de la conversación, si no sabes algún dato responde con "No lo sé"
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        human_template ="{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt,human_message_prompt])
        chain =load_summarize_chain(llm,
                                    chain_type="map_reduce",
                                    map_prompt=chat_prompt
                                    )
        
        output =chain.run({
                "input_documents": texts,
        })
        print(output)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return output


def resumen_opcion_multiple(user,user_selection):
    with get_openai_callback() as cb:
        storage_client = storage.bucket("samai-b9f36.appspot.com")
        folder_name = f"{user}/{user}_workflow"
        file_name = f"{folder_name}/test_data_{user}.txt"

        # Crear el directorio de destino si no existe
        os.makedirs(folder_name, exist_ok=True)

        blob = storage_client.blob(file_name)
        archivo_destino = f"./{file_name}"
        blob.download_to_filename(archivo_destino)

        with open(archivo_destino, "r") as f:
            contenido = f.read()
            print(contenido)
        

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=250)
        texts = text_splitter.create_documents([contenido])

        llm = ChatOpenAI(temperature=0,openai_api_key=openai.api_key)

        summary_output_options = {
                'one_sentence': """
                -Solo un enunciado, no mas de 10 palabras.
                """,
                'bullet_points': """
                -Bullet point format
                -Separate each bullet point with a new line
                -Each bullet point should be concise
                """,
                'short':"""
                -A few short sentences
                -Do not go longer than 4-5 sentences
                """,
                'long': """
                -A verbose summary
                -You may do a few paragraphs to descript the transcript if needed
                """
        }

        template = """
#         Eres un análista de conversaciones para una empresa de marketing.
#         Tu Compañía es lo que en la conversación viene como "role":"asssistant" y el usuario como "role":"user".
#         Vas a analizar lo que el "role":"user" comenta.
#         Tu meta es escribir un resumen de los comentarios del usuario y abstraer la información personal que el otorgue.
#         Define un sentimiento que refleje los comentarios del usuario.
#         Es importante que resaltes los puntos mas importantes de la conversación.
#         No respondas con nada fuera de la conversación, si no sabes algún dato responde con "No lo sé".
#         Encuentra el nombre del usuario, su correo electrónico y su lugar de residencia, si algo falta especifícalo
#         """
        
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

        human_template ="{text}"
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map,human_message_prompt_map])

        template = """
#         Eres un análista de conversaciones para una empresa de marketing.
#         Tu Compañía es lo que en la conversación viene como "role":"asssistant" y el usuario como "role":"user".
#         Vas a analizar lo que el "role":"user" comenta.
#         Tu meta es escribir un resumen de los comentarios del usuario y abstraer la información personal que el otorgue.
#         Define un sentimiento que refleje los comentarios del usuario.
#         Es importante que resaltes los puntos mas importantes de la conversación.
#         No respondas con nada fuera de la conversación, si no sabes algún dato responde con "No lo sé".
#         Encuentra y muestra el nombre del usuario, su correo electrónico y su lugar de residencia, si algo falta especifícalo.
#         Muestra los datos del usuario de la siguiente manera:
#         -Nombre : Nombre del usuario
#          -Email: Correo electrónico del usuario
#           -Lugar de residencia: Lugar de residencia
#         Ejemplo:
#         -Nombre : Orlando
#         -Email: orlando.gmail.com    
#          -Lugar de residencia: puebla
        

        Responde con el siguiente formato:
        {output_format}
        """
        
        system_message_prompt_combine = SystemMessagePromptTemplate.from_template(template)

        human_template ="{text}"
        human_message_prompt_combine = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_combine,human_message_prompt_combine])
        chain =load_summarize_chain(llm,
                                    chain_type="map_reduce",
                                    map_prompt=chat_prompt_map,
                                    combine_prompt =chat_prompt_combine,
                                    verbose=True
                                    )
        user_selection = user_selection
        
        output =chain.run({
                "input_documents": texts,
                "output_format": summary_output_options[user_selection]
        })
        print(output)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return output


def vector_index(user):
        storage_client = storage.bucket("samai-b9f36.appspot.com")
        folder_name = f"{user}/{user}_workflow"
        file_name = f"{folder_name}/test_data_{user}.txt"

        # Crear el directorio de destino si no existe
        os.makedirs(folder_name, exist_ok=True)

        blob = storage_client.blob(file_name)
        archivo_destino = f"./{file_name}"
        blob.download_to_filename(archivo_destino)

        documents = SimpleDirectoryReader(folder_name).load_data()
        index = GPTVectorStoreIndex.from_documents(documents)
        print(index)
        index.storage_context.persist()
        # Ruta local de la carpeta "storage"
        ruta_carpeta_local = f"storage"

        # Ruta de la carpeta en Firebase Storage
        ruta_carpeta_destino = f"{user}/storage_{user}"

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
        return "done"

def pregunta_data(user,question):
    storage_client = storage.bucket("samai-b9f36.appspot.com")

    prefix = f"{user}/storage_{user}"

    local_folder = f"./storage_{user}"
    os.makedirs(local_folder,exist_ok=True)

    blobs = storage_client.list_blobs(prefix=prefix)

    for blob in blobs:
        nombre_archivo =blob.name.split("/")[-1]

        archivo_local = os.path.join(local_folder,nombre_archivo)
        blob.download_to_filename(archivo_local)

    storage_context = StorageContext.from_defaults(persist_dir=f"./{local_folder}")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    x = question
    response = query_engine.query(x)
    print(response)
    return response

def borrar_contenido(user):
    storage_client = storage.bucket("samai-b9f36.appspot.com")

    # Ejemplo de uso
    workflow_folder_path = f'{user}/{user}_workflow'
    data_folder_path = f'{user}/storage_user'
    local_storage_folder = "./storage"
    local_storage_folder2 = f"./storage_{user}"
    local_storage_folder3 = f"./{user}_workflow"
    local_storage_folder4 = f"./{user}"

    try:
        # Obtener la lista de archivos en la carpeta de workflow
        workflow_blobs = storage_client.list_blobs(prefix=workflow_folder_path)
        for blob in workflow_blobs:
            if not blob.name.endswith('/'):  # Verificar si el objeto es un archivo y no una carpeta
                blob.delete()

        # Obtener la lista de archivos en la carpeta de data_storage
        data_blobs = storage_client.list_blobs(prefix=data_folder_path)
        for blob in data_blobs:
            if not blob.name.endswith('/'):  # Verificar si el objeto es un archivo y no una carpeta
                blob.delete()
        
           # Borrar el contenido de la carpeta local "storage"
        shutil.rmtree(local_storage_folder)
        shutil.rmtree(local_storage_folder2)
        shutil.rmtree(local_storage_folder3)
        shutil.rmtree(local_storage_folder4)

        print('Contenido de las carpetas eliminado exitosamente')
    except Exception as e:
        print('Error al eliminar el contenido de las carpetas:', str(e))


