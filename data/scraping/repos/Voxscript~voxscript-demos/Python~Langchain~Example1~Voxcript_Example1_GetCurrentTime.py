import os
os.environ["OPENAI_API_KEY"] = "your-openapi-key-here"
os.environ["VOXSCRIPT_APIKEY"] = "your-voxscript-key-here" #Get at https://voxscript-api.awt.icu

import yaml
import requests
from langchain.chat_models import ChatOpenAI
from langchain.requests import RequestsWrapper
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec

# Initialize ChatGPT-4
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Download and parse the Voxscript API specification
voxscript_api_url = 'https://voxscript.awt.icu/swagger/v1/swagger.yaml'
response = requests.get(voxscript_api_url)
raw_voxscript_api_spec = yaml.load(response.content, Loader=yaml.Loader)

#Once our API spec is finalized, you won't need to add the server to the existing yaml..

server_to_add = [
    {"url": "http://voxscript.awt.icu"},
]
# Add the server block only if it doesn't exist, once our API spec is finalized this will be present from the swagger.yaml
if 'servers' not in raw_voxscript_api_spec:
    print("Adding new servers block.")
    raw_voxscript_api_spec['servers'] = server_to_add
else:
    print("Servers block already exists. No changes made.")

# End Hack :)

# Construct authentication headers for Voxscript
def construct_voxscript_auth_headers():
    # Placeholder for actual authentication implementation
    return {"Authorization": "Bearer " + os.environ['VOXSCRIPT_APIKEY']}

# Create requests wrapper with Voxscript headers
headers = construct_voxscript_auth_headers()
requests_wrapper = RequestsWrapper(headers=headers)

# Create Voxscript API agent
voxscript_api_spec = reduce_openapi_spec(raw_voxscript_api_spec)  # Assuming this is the parsed spec
voxscript_agent = planner.create_openapi_agent(voxscript_api_spec, requests_wrapper, llm)

# Example usage
user_query = "What is the current time?"
response = voxscript_agent.run(user_query)
print(response)