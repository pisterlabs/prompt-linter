def get_cell1_content(db, asnc, func):
    db1_content = """
chroma_client = chromadb.Client()

embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002",
)

collection = chroma_client.create_collection(name="", embedding_function=embedder)"""

    imports = "from aitemplates import Message, ChatSequence"
    
    if asnc or func:
        imports += ", ChatConversation"
        
    if asnc:
        imports += ", async_create_chat_completion"
    else:
        imports += ", create_chat_completion"
    if func:
        imports += ", FunctionPair, Functions"
        
    if db:
        imports += """
import os
import chromadb
from chromadb.utils import embedding_functions"""
        
    
    cell1_content = f"""\
{imports}
import openai
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")
{db1_content if db else ""}"""
    return cell1_content


def get_cell2_content(db, asnc, func):
    additional = ""
    
    if asnc or func:
        additional+= """
sequence2 = ChatSequence([user_query_msg])
chat = ChatConversation([sequence1, sequence2])"""
        
    cell2_content = f"""\
system_prompt=""
system_prompt_msg = Message("system", system_prompt)

description = ""
description_msg = Message("system", description)

user_query=""
user_query_msg = Message("user", user_query)

sequence1 = ChatSequence([system_prompt_msg, description_msg, user_query_msg])
{additional}"""
    return cell2_content


def get_cell3_content(db, asnc, func):
    additional = ""
    
    if func:
        additional+="""\
from aitemplates import FunctionDef

stock_price = FunctionDef(
    name="func_name",
    description="Use this function to get the price of a stock.",
    parameters={
        "property1": FunctionDef.ParameterSpec(
            name="property1",
            type="string",
            description="",
            required=True
        )
    }
)
    
def func(property1: str) -> float:
    return 0.0
    
# match the description to the function
function_pair1 = FunctionPair(func_desc, func)

# add Functions dataclass for easy access
functions_available = Functions([function_pair1])
"""
        
    if asnc:
        additional+=f"\nasync_response = await async_create_chat_completion(chat, keep_order=True)"
    else:
        additional+=f"\ncompletion = create_chat_completion(sequence1{', functions=functions_available, auto_call_func=True' if func else ''})"

    cell3_content = f"""\
{additional}"""
    return cell3_content

def get_cell4_content(db, asnc, func):
    additional = ""
    
    if asnc and db:
        additional+="""\
collection.add(
    documents=[*async_response.get_last_responses()], # takes param of num of responses, or call .get_last_responses(all=True)
    metadatas=[{"": ""}, {"": ""}],
    ids=["", ""]
)"""
    elif db:
        additional += """
collection.add(
    documents=[completion],
    metadatas=[{"": ""}],
    ids=[""]
)"""

    cell4_content = f"""\
{additional}
{"async_response.display_conversation()" if asnc else "completion"}"""
    return cell4_content

def get_cell5_content(db, asnc, func):
    cell5_content = f"""\
from aitemplates import SingleApiManager

api_manager = SingleApiManager()
api_manager.total_cost"""
    return cell5_content