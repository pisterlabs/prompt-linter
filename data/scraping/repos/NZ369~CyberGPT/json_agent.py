import os
import boto3
import pandas as pd
from llms.azure_llms import create_llm
from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_types import AgentType

llm = create_llm(temp=0)
llm.request_timeout=15

print("Creating JSON Agent.")

# Waiting on Secret Key info for Amazon.
def download_files(bucket_name="team5.2-mitre", data_dir="data", files=[]):
    s3_client = boto3.client('s3')
    os.makedirs(data_dir, exist_ok=True)
    for fn in files:
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            print(f"Downloading {fn}")
            s3_client.download_file(bucket_name, fn, fp)

download_files(files=["apt1.json"])

combined_data = os.path.join("data", "apt1.json")
data = pd.read_json(combined_data)
df = pd.json_normalize(data)

json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

stix_json_agent = create_json_agent(
    llm,
    toolkit=json_toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)
print("Finished Creating JSON Agent.")

#stix_json_agent.run("What type of threat actor is SuperHard?")
