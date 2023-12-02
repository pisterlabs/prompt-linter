import os
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain.document_loaders import TextLoader
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.http import models
import requests

from agent.utils.utility import replace_multiple_whitespaces
#from agent.backend.qdrant_service import get_qdrant_client

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOllama                                

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

channeling_system_message = """Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT"""

#q_and_a_system_message = """Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT"""

#q_and_a_system_message = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in the language you got asked."""

q_and_a_system_message = os.getenv("Q_A_SYSTEM_MESSAGE",default="Du bist ein ehrlicher, respektvoller und ehrlicher Assistent. Zur Beantwortung der Frage nutzt du nur den Text, welcher zwischen <INPUT> und </INPUT> steht! Findest du keine Informationen im bereitgestellten Text, so antwortest du mit 'Ich habe dazu keine Informationen'")



@load_config(location="config/db.yml")
def get_db_connection(cfg: DictConfig) -> Qdrant:
    """get_db_connection initializes the connection to the Qdrant db.

    :param cfg: OmegaConf configuration
    :type cfg: DictConfig
    :return: Qdrant DB connection
    :rtype: Qdrant
    """
    embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL) #OllamaEmbeddings(base_url="http://" + OLLAMA_URL + ":" + OLLAMA_PORT, model=OLLAMA_MODEL)
    qdrant_client = QdrantClient(os.getenv("QDRANT_URL",cfg.qdrant.url), port=os.getenv("QDRANT_PORT",cfg.qdrant.port), api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    try: 
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_llama2)  
        
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=cfg.qdrant.collection_name_llama2,
            vectors_config=models.VectorParams(size=len(embedding.embed_query("Test text")), distance=models.Distance.COSINE),
        )
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_llama2} created.")
    vector_db = Qdrant(client=qdrant_client, collection_name=cfg.qdrant.collection_name_llama2, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB Connection.")
    return vector_db



#just for tests or future summaries when the prompt gets too long
@load_config(location="config/ai/llama2.yml")
def summarize_text_ollama(text: str, cfg: DictConfig) -> str:
    """Summarizes the given text using the llama2 API.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summary of the text.
    """
    prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-summarization.j2", text=text, language="de")

    ollama_model = f"{os.environ.get('OLLAMA_MODEL')}:{os.environ.get('OLLAMA_MODEL_VERSION')}" if os.environ.get('OLLAMA_MODEL_VERSION') else os.environ.get('OLLAMA_MODEL')

    llm = ChatOllama(
    base_url="http://" + OLLAMA_URL + ":" + OLLAMA_PORT,
    model=ollama_model,
    verbose=True
    )

    response = llm(prompt)

    return response


def embedd_documents_ollama(dir: str) -> None:
    """Embeds the documents in the given directory in the llama2 database.

    This method uses the Directory Loader for PDFs and the PyPDFLoader to load the documents.
    The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

    Args:
        dir (str): The directory containing the PDFs to embed.

    Returns:
        None
    """
    vector_db = get_db_connection()

    logger.info(f"Logged directory:  {dir}")
    loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
    length_function = len
    splitter = CharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        length_function=length_function,
    )
    docs = loader.load_and_split(splitter)
    logger.info(f"Loaded {len(docs)} documents.")
    text_list = [doc.page_content for doc in docs]
    metadata_list = [doc.metadata for doc in docs]
    vector_db.add_texts(texts=text_list, metadatas=metadata_list)
    logger.info("SUCCESS: Texts embedded.")


def embedd_text_ollama(text: str, file_name: str, seperator: str) -> None:
    """Embeds the given text in the llama2 database.

    Args:
        text (str): The text to be embedded.


    Returns:
        None
    """
    vector_db = get_db_connection()

    # split the text at the seperator
    text_list: List = text.split(seperator)

    # check if first and last element are empty
    if not text_list[0]:
        text_list.pop(0)
    if not text_list[-1]:
        text_list.pop(-1)

    metadata = file_name
    # add _ and an incrementing number to the metadata
    metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]

    vector_db.add_texts(texts=text_list, metadatas=metadata_list)
    logger.info("SUCCESS: Text embedded.")


def embedd_text_files_ollama(folder: str, seperator: str) -> None:
    """Embeds text files in the llama2 database.

    Args:
        folder (str): The folder containing the text files to embed.
        seperator (str): The seperator to use when splitting the text into chunks.

    Returns:
        None
    """
    vector_db = get_db_connection()

    # iterate over the files in the folder
    for file in os.listdir(folder):
        # check if the file is a .txt or .md file
        if not file.endswith((".txt", ".md")):
            continue

        # read the text from the file
        with open(os.path.join(folder, file)) as f:
            text = f.read()

        text_list: List = text.split(seperator)

        # check if first and last element are empty
        if not text_list[0]:
            text_list.pop(0)
        if not text_list[-1]:
            text_list.pop(-1)

        # ensure that the text is not empty
        if not text_list:
            raise ValueError("Text is empty.")

        logger.info(f"Loaded {len(text_list)} documents.")
        # get the name of the file
        metadata = os.path.splitext(file)[0]
        # add _ and an incrementing number to the metadata
        metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]
        vector_db.add_texts(texts=text_list, metadatas=metadata_list)

    logger.info("SUCCESS: Text embedded.")
    
 
def search_documents_ollama(query: str, amount: int, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
    """Searches the documents in the Qdrant DB with a specific query.

    Args:
        query (str): The question for which documents should be searched.

    Returns:
        List[Tuple[Document, float]]: A list of search results, where each result is a tuple
        containing a Document object and a float score.
    """
    vector_db = get_db_connection()

    docs = vector_db.similarity_search_with_score(query, k=amount)
    

    logger.info("SUCCESS: Documents found after similarity_search_with_score.")
    #logger.info(f"These are the docs found after similarity_search_with_score: {docs}")


    if os.environ.get('ACTIVATE_RERANKER') == "True":

        embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        filtered_docs = [t[0] for t in docs]
        retriever = vector_db.from_documents(filtered_docs, embedding, api_key=os.environ.get('QDRANT_API_KEY'), url=os.environ.get('QDRANT_URL')).as_retriever()

        #cohere multi lang rerank model only supports none-english documents
        rerank_compressor = CohereRerank(user_agent="my-app", model="rerank-multilingual-v2.0", top_n=1)
        splitter = CharacterTextSplitter(chunk_size=120, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
        relevant_filter = EmbeddingsFilter(embeddings=embedding)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter, rerank_compressor]
        )
        compression_retriever1 = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

        compressed_docs = compression_retriever1.get_relevant_documents(query)

        for docu in compressed_docs:
            logger.info(f"These are the docs found after reranking: {replace_multiple_whitespaces(docu.page_content)}") 

        return compressed_docs
    else:
        filtered_docs = [t[0] for t in docs]
        return filtered_docs

@load_config(location="config/ai/llama2.yml")
def send_chat_completion_ollama(text: str, query: str, cfg: DictConfig, conversation_type: str, messages: any) -> str:
    """Sent completion request to llama2 API.

    Args:
        text (str): The text on which the completion should be based.
        query (str): The query for the completion.
        cfg (DictConfig):

    Returns:
        str: Response from the llama2 API.
    """

    #fill the prompt if not q and a then it will use channeling with the documents
    if conversation_type == "CHANNELING":
        prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-channeling.j2", text=text, query=query, system=channeling_system_message, language="de")
    else:
        prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-qa.j2", text=text, query=query, system=q_and_a_system_message, language="de")
    messages.append({"role": "user", "content": prompt})
    logger.info(f"DEBUG: This is the filled prompt before request: {prompt}")

    
    ollama_model = f"{os.environ.get('OLLAMA_MODEL')}:{os.environ.get('OLLAMA_MODEL_VERSION')}" if os.environ.get('OLLAMA_MODEL_VERSION') else os.environ.get('OLLAMA_MODEL')

    # llm = ChatOllama(
    # base_url="http://" + OLLAMA_URL + ":" + OLLAMA_PORT,
    # model=ollama_model,
    # verbose=True
    # )

    messagesBaseFormat: List[BaseMessage] = [HumanMessage(content=m["content"], additional_kwargs={}) if m["role"] == "user"
                      else AIMessage(content=m["content"], additional_kwargs={}) if m["role"] == "assistant"
                      else SystemMessage(content=m["content"], additional_kwargs={})
                      for m in messages]

    raw_mode = os.environ.get('OLLAMA_RAW_MODE', default = "False").lower() in ['true']


    response = generate_ollamaRequest(url_ollama_generateEndpoint="http://ollama.one-cx.org:80/api/generate",
                                      model=os.environ.get('OLLAMA_MODEL'),
                                      full_prompt=prompt)

    # response = llm.generate(
    #     messages=[messagesBaseFormat],        
    # )
    
    logger.info(f"DEBUG: response: {response}")
    return response

def chat_ollama(documents: list[tuple[LangchainDocument, float]], messages: any, query: str, conversation_type: str, summarization: bool = False) -> Tuple[str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
    """QA takes a list of documents and returns a list of answers.

    Args:
        documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
        query (str): The query to ask.
        summarization (bool, optional): Whether to use summarization. Defaults to False.

    Returns:
        Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
    """
    text = ""
    if conversation_type == "Q_AND_A":
        # if the list of documents contains only one document extract the text directly
        if len(documents) == 1:
            texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
            text = " ".join(texts)
            meta_data = [doc.metadata for doc in documents]

        else:
            # extract the text from the documents
            texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
            if summarization:
                # call summarization
                logger.info(f"woudl call a summary here")

            else:
                # combine the texts to one text
                text = " ".join(texts)
            meta_data = [doc.metadata for doc in documents]
    else:
        # if the list of documents contains only one document extract the text directly
        if len(documents) == 1:
            texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
            text = " ".join(texts)
            meta_data = [doc.metadata for doc in documents]

        else:
            # extract the text from the documents
            texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
            if summarization:
                # call summarization
                logger.info(f"woudl call a summary here")

            else:
                # combine the texts to one text
                text = " ".join(texts)
            meta_data = [doc.metadata for doc in documents]
    
    answer=""
    try:
        # call the gpt api

        answer = send_chat_completion_ollama(text=text, query=query, conversation_type=conversation_type, messages=messages)

    except ValueError as e:
        #when prompt is too large it can be implemented here
        logger.debug("DEBUG: Error found.")
        logger.error(e)
        answer = "Error"
    logger.debug(f"LLM response: {answer}")
    
    return answer, meta_data


def generate_ollamaRequest(url_ollama_generateEndpoint: str, model: str, full_prompt: str):
    url = url_ollama_generateEndpoint
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "template": full_prompt,
        "stream": False,
        "options": {"stop": ["<|im_start|>", "<|im_end|>"]}
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        logger.debug("Request was successful!")
        logger.debug("Response:")
        logger.debug(response.json()["response"])
    else:
        logger.debug(f"Error {response.status_code}: {response.text}")

    return response.json()["response"]



if __name__ == "__main__":
    print("Test")


