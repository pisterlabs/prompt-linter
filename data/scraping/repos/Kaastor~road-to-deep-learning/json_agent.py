import os
from datetime import datetime, timedelta

from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.llms.openai import OpenAI
from langchain.tools.json.tool import JsonSpec
from pymongo import MongoClient

os.environ["OPENAI_API_KEY"] = ''

uri = 'mongodb://127.0.0.1'
_mongo_client = MongoClient(uri)
db = _mongo_client.get_database("C03i2p9bk-drive")
coll = db.get_collection("file_meta")

current_date = datetime.now()
six_months_ago = current_date - timedelta(days=35)

result = coll.find_one({"owner": {"$eq": "susan@generalaudittool.com"}})

json_spec = JsonSpec(dict_=result)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
)

response = json_agent_executor.run(
    "What kind of file types file owner called susan@generalaudittool.com has? "
    "Show me detailed summary across all jsons in supplied data."
)

print(response)
