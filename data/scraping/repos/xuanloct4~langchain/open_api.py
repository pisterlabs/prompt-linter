import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

import os, yaml
# %cd openapi
# !wget https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
# !mv openapi.yaml openai_openapi.yaml
# !wget https://www.klarna.com/us/shopping/public/openai/v0/api-docs
# !mv api-docs klarna_openapi.yaml
# !wget https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml
# !mv openapi.yaml spotify_openapi.yaml

from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec

with open("openapi/openai_openapi.yaml") as f:
    raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
    
with open("openapi/klarna_openapi.yaml") as f:
    raw_klarna_api_spec = yaml.load(f, Loader=yaml.Loader)
klarna_api_spec = reduce_openapi_spec(raw_klarna_api_spec)

with open("openapi/spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)

import spotipy.util as util
from langchain.requests import RequestsWrapper

def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = util.prompt_for_user_token(scope=','.join(scopes))
    return {
        'Authorization': f'Bearer {access_token}'
    }

# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)

endpoints = [
    (route, operation)
    for route, operations in raw_spotify_api_spec["paths"].items()
    for operation in operations
    if operation in ["get", "post"]
]
len(endpoints)

import tiktoken
enc = tiktoken.encoding_for_model('text-davinci-003')
def count_tokens(s): return len(enc.encode(s))

count_tokens(yaml.dump(raw_spotify_api_spec))

from langchain.agents.agent_toolkits.openapi import planner
# from langchain.llms.openai import OpenAI
# llm = OpenAI(model_name="gpt-4", temperature=0.0)
spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
user_query = "make me a playlist with the first song from kind of blue. call it machine blues."
spotify_agent.run(user_query)

user_query = "give me a song I'd like, make it blues-ey"
spotify_agent.run(user_query)

headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}
openai_requests_wrapper=RequestsWrapper(headers=headers)
# # Meta!
# llm = OpenAI(model_name="gpt-4", temperature=0.25)
openai_agent = planner.create_openapi_agent(openai_api_spec, openai_requests_wrapper, llm)
user_query = "generate a short piece of advice"
openai_agent.run(user_query)


from langchain.agents import create_openapi_agent
from langchain.agents.agent_toolkits import OpenAPIToolkit
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
with open("openapi/openai_openapi.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec=JsonSpec(dict_=data, max_value_length=4000)


openapi_toolkit = OpenAPIToolkit.from_llm(llm, json_spec, openai_requests_wrapper, verbose=True)
openapi_agent_executor = create_openapi_agent(
    llm=llm,
    toolkit=openapi_toolkit,
    verbose=True
)

openapi_agent_executor.run("Make a post request to openai /completions. The prompt should be 'tell me a joke.'")
