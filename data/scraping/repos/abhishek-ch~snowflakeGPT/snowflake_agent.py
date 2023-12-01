import os
import openai
from langchain import OpenAI, PromptTemplate, SQLDatabase, SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI, OpenAI
from langchain.llms.openai import BaseOpenAI
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from snowgpt.prompt import *
from snowgpt.db.redis_vectordb import create

gpt_models_dict = {"gpt-35-turbo":"gpt-3.5-turbo",
                   "gpt-4_8k_ascent":"gpt-4-0314",
                   "gpt-4_32k_ascent":"gpt-4-32k-0314"}

def get_model(gpt_model_name:str) -> BaseOpenAI:
    callbacks = [StreamingStdOutCallbackHandler()]
    model_type = os.environ.get("MODEL_TYPE")

    match model_type:
        case "azure":
            return ChatOpenAI(engine=gpt_model_name, model_name=gpt_models_dict.get(gpt_model_name), temperature=0.2)
        case "openai":
            return OpenAI(temperature=0.7)
        case "LlamaCpp":
            return LlamaCpp(model_path=os.environ["LLAMA_EMBEDDINGS_MODEL"], n_ctx=21000, temperature=0.8, n_threads=6, callbacks=callbacks)
        case "GPT4All":
            return GPT4All(model=os.environ["MODEL_PATH"], n_ctx=1048, backend='gptj')
        case _default:
            print(f"Model {model_type} is not supported") 
            exit;

def verify_prompt(example_input:str,model:str="gpt-35-turbo") -> None:
    # llm = get_model(model)
    prompt = get_prompt_template()
    print(
        prompt.format(
            input=example_input, chat_history=None
        ))

def query_database(query:str):
    vector_db = create()
    docs = vector_db.similarity_search(query)
    # print(docs[0].page_content)
    if docs:
        return docs[0].page_content
    return ""

def execute(query: str, model:str="gpt-35-turbo") -> str:
    assert os.environ.get("SNOWFLAKE_URL") is not None,"Snowflake URL must be configured as Env variable SNOWFLAKE_URL"
    result = query_database(query)
    snowflake_prompt_template = get_prompt_template(result)
    database = SQLDatabase.from_uri(os.environ.get("SNOWFLAKE_URL"),sample_rows_in_table_info=0)
    VECTORDB_DECIDER_TEMPLATE = generate_vectordb_template(result[0])
    seq_chain = SQLDatabaseSequentialChain.from_llm(get_model(model), database, verbose=True,
                                                query_prompt=snowflake_prompt_template,
                                                decider_prompt=VECTORDB_DECIDER_TEMPLATE,
                                                return_intermediate_steps=True)

    print(seq_chain.decider_chain)
    result = seq_chain(query)
    print(result)