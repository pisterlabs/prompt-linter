from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from configer import ConfigLoader

configer = ConfigLoader()
OPENAI_API_KEY = configer.get_api_key()
OPENAI_API_BASE = configer.get_api_base()

embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)

def search_personal_vector_db(query: str):
    """Let GPT return a modified user query to search the local database
    and answer local database information.
    """
    db = FAISS.load_local(
        configer.config["system"]["vector_db"]["store_path"], 
        embeddings=embedding_model
    )
    docs = db.similarity_search(query)
    return docs


FUNCTION_LIB = {
    "search_personal_vector_db": search_personal_vector_db,
}

FUNCTION = [
    {
        "name": "search_personal_vector_db",
        "description": "Let GPT return a modified user query to search the local database \
    and answer local database information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User input query, GPT can modified for better similarity search",
                },
            },
            "required": ["query"]
        },
    },
]
