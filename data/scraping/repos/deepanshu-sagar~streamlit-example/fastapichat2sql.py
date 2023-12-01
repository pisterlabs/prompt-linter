from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from typing import Any, Dict
from sql_connector import SqlConnector
from langchain.llms import OpenAI
from collections import namedtuple

class Input(BaseModel):
    prompt: str
    api_key: str

class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)

class ChatSql:

    def __init__(self) -> None:
        conf_path: str = get_final_path(1, ["conf.json"])
        self.conf: Dict[str, Any] = read_json(conf_path)
        self.llm: object = OpenAI(openai_api_key=self.conf["OPEN_AI_KEY"])
        self.info: str = str(read_json(get_final_path(1, ["info.json"])))
        self.data: str = str(read_txt(get_final_path(1, ["data.txt"])))

    def prompt_to_query(self, prompt: str) -> Dict[str, str]:
        info = self.info
        data= self.data
        template = """
        Your mission is convert SQL query from given {prompt}. Use following database information for this purpose (info key is a database column name and info value is explanation) : {info} .  along with this i am sharing some sample data from this table :  {data}
        --------
        Put your query in the  JSON structure with key name is 'query'
        """
        pr_ = PromptTemplate(input_variables=["prompt", "info"], template=template)
        final_prompt = pr_.format(
            prompt=prompt,
            info=info,
            data=data
        )
        gpt_query: Dict["str", "str"] = json.loads(self.llm(final_prompt))

        return gpt_query

    def query_to_result(self, gpt_query: Dict[str, str]) -> str:
        raw_res: str = SqlConnector(query=gpt_query["query"]).query
        return raw_res

    def raw_result_to_processed(self, raw_result: str) -> str:
        res_processing_template = """
        Your mission is convert database result to meaningful sentences. Here is the database result: {database_result}. validate the query before responding.
        """
        db_pr = PromptTemplate(
            input_variables=["database_result"], template=res_processing_template
        )
        final_prompt = db_pr.format(database_result=raw_result)
        procesed_result: str = self.llm(final_prompt)

        return procesed_result

def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def read_txt(file_path: str) -> str:
    with open(file_path, "r") as file:
        data = file.read()
    return data

def save_json(file_path: str, data: Dict[str, Any]):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def get_final_path(path_type: int, paths: list) -> str:
    current_directory = os.getcwd()
    final_path = os.path.join(current_directory, *paths)
    return final_path

app = FastAPI()
"""
curl --location --request POST 'http://localhost:5002/process' \
--header 'api_key: <>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "prompt": "How many records have less than 2 schain nodes?",
  "api_key":"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"
}'
"""
@app.post("/process")
async def process(input: Input):
    csql = ChatSql()
    csql.llm = OpenAI(openai_api_key=input.api_key)  # update the API Key
    query = csql.prompt_to_query(input.prompt)
    result = {"Generated SQL Query": query["query"]}
    return result
