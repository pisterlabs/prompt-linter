from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Redis
from langchain.tools import tool
import os


class Question(BaseModel):
    __root__: str


EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_SCHEMA = os.path.join(os.path.dirname(__file__), "schema.yml")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# Connect to pre-loaded vectorstore
# run the ingest.py script to populate this


def check_index(patient_id):
    try:
        vectorstore = Redis.from_existing_index(
            embedding=embedder, index_name=patient_id, schema=INDEX_SCHEMA, redis_url=REDIS_URL
        )
        return vectorstore.as_retriever(search_type="mmr")
    except Exception as e:
        if e == ValueError:
            return False
        else:
            raise e

# Usage: tools = [medpromt.chains.get_chain]
@tool(args_schema=Question)
def get_rag_chain(patient_id: str, **kwargs):
    """
    Returns a chain that can be used to answer a question based on a patient's medical record.

    Args:
        patient_id (str): The id of the patient to search for.
        llm (LangModel): The language model to use to answer the question.
        prompt (ChatPromptTemplate): The prompt to use to ask the question if available.
        output_parser (OutputParser): The output parser to use to parse the answer if available.
    """
    llm = kwargs.get("llm", None)
    retriever = check_index(patient_id)
    if not retriever:
        raise ValueError("No index found.")
    # Define our prompt
    template = """
    TBD

    Context:
    ---------
    {context}

    ---------
    Question: {question}
    ---------

    Answer:
    """

    _prompt = ChatPromptTemplate.from_template(template)
    prompt = kwargs.get("prompt", _prompt)
    output_parser = kwargs.get("output_parser", StrOutputParser())

    if not llm:
        raise ValueError("No language model provided.")
    # RAG Chain
    # model = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | output_parser
    ).with_types(input_type=Question)
    return chain