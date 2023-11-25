from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from glob import glob
from tqdm import tqdm
import yaml

def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

def load_embeddings(model_name=config["embeddings"]["name"],
                    model_kwargs = {'device': config["embeddings"]["device"]}):
    # load_dotenv()
    return OpenAIEmbeddings()

def load_documents(directory : str):
    """Loads all documents from a directory and returns a list of Document objects
    args: directory format = directory/
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = config["TextSplitter"]["chunk_size"], 
                                                   chunk_overlap = config["TextSplitter"]["chunk_overlap"])
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        loader = PyPDFLoader(item_path)
        documents.extend(loader.load_and_split(text_splitter=text_splitter))

    return documents

def load_db(embedding_function, 
            save_path=config["chroma_indexstore"]["save_path"], 
            index_name=config["chroma_indexstore"]["index_name"]):
    db = Chroma(persist_directory=save_path, embedding_function=embedding_function)
    return db

def save_db(db, 
            save_path=config["chroma_indexstore"]["save_path"], 
            index_name=config["chroma_indexstore"]["index_name"]):
    db.persist()
    print("Saved db to " + save_path + index_name)

def condense_question_prompt():
    CONDENSE_PROMPT = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question and respond in english.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    return PromptTemplate(
        input_variables=['chat_history','question'],
        template=CONDENSE_PROMPT
    )

def question_answer_prompt():
    QUESTION_ANSWER_PROMPT_DOCUMENT_CHAT = """You are a helpful and courteous support representative working for an insurance company. 
    Use the following pieces of context to answer the question at the end.
    If the question is not related to the context, politely respond that you are tought to only answer questions that are related to the context.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer. 
    Try to make the title for every answer if it is possible. Answer in markdown.
    Make sure that your answer is always in Markdown.
    {context}
    Question: {question}
    Answer in HTML format:"""

    return PromptTemplate(
        input_variables=['context', 'question'],
        template=QUESTION_ANSWER_PROMPT_DOCUMENT_CHAT
    )

def get_chat_history(inputs) -> str:
    res = []
    # print('am here =====>')
    print(inputs)
    # if (len(inputs) > 0):
    #     human, ai = inputs
    #     res.append(f"Human:{human}\nAI:{ai}")
    for i in range(0, len(inputs), 2):
        human = inputs[i]
        ai = inputs[i + 1]
        res.append(f"Human:{human}\nAI:{ai}")
    # for human, ai in inputs:
    #     res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)