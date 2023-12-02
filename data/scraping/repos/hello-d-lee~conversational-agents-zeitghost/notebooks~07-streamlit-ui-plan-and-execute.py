from os import system
from pathlib import Path

import sys
import os
sys.path.append("..")

import streamlit as st
from langchain import SQLDatabase
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, SQLDatabaseChain
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings




from zeitghost.agents.LangchainAgent import LangchainAgent

from streamlit_agent.callbacks.capturing_callback_handler import playback_callbacks
from streamlit_agent.clear_results import with_clear_container

st.set_page_config(
    page_title="Google Langchain Agents", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed"
)
"# 🦜🔗 Langchain for Google Palm"

ACTOR_PREFIX             = "ggl"
VERSION                  = 'v1'
PROJECT_ID               = 'cpg-cdp'
BUCKET_NAME              = f'zghost-{ACTOR_PREFIX}-{VERSION}-{PROJECT_ID}'
BUCKET_URI               = f'gs://{BUCKET_NAME}'


###HARDCODED VALUES BELOW - TODO UPDATE LATER

PROJECT_ID               = "cpg-cdp"
PROJECT_NUM              = "939655404703"
LOCATION                 = "us-central1"
REGION                   = "us-central1"
BQ_LOCATION              = "US"
VPC_NETWORK_NAME         = "genai-haystack-vpc"
CREATE_NEW_ASSETS        = "True"
VERSION                  = "v1"
ACTOR_NAME               = "google"
ACTOR_CATEGORY           = "technology"
BUCKET_NAME              = "zghost-ggl-v1-cpg-cdp"
EMBEDDING_DIR_BUCKET     = "zghost-ggl-v1-cpg-cdp-emd-dir"
BUCKET_URI               = "gs://zghost-ggl-v1-cpg-cdp"
EMBEDDING_DIR_BUCKET_URI = "gs://zghost-ggl-v1-cpg-cdp-emd-dir"
VPC_NETWORK_FULL         = "projects/939655404703/global/networks/me-network"
ME_INDEX_NAME            = "vectorstore_ggl_v1"
ME_INDEX_ENDPOINT_NAME   = "vectorstore_ggl_v1_endpoint"
ME_DIMENSIONS            = "768"
MY_BQ_DATASET            = "zghost_ggl_v1"
MY_BQ_TRENDS_DATASET     = "zghost_ggl_v1_trends"


#TODO - this works fine from a notebook but getting UNKNOWN errors when trying to access ME from a signed-in env (for user)
# from zeitghost.vertex.Embeddings import VertexEmbeddings

from zeitghost.vertex.MatchingEngineCRUD import MatchingEngineCRUD
from zeitghost.vertex.MatchingEngineVectorstore import MatchingEngineVectorStore

# Google Cloud
# from google.cloud import aiplatform as vertex_ai
# from google.cloud import storage
# from google.cloud import bigquery


#Instantiate Google cloud SDK clients
# storage_client = storage.Client(project=PROJECT_ID)

## Instantiate the Vertex AI resources, Agents, and Tools
mengine = MatchingEngineCRUD(
    project_id=PROJECT_ID 
    , project_num=PROJECT_NUM
    , region=LOCATION 
    , index_name=ME_INDEX_NAME
    , vpc_network_name=VPC_NETWORK_FULL
)

ME_INDEX_RESOURCE_NAME, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
ME_INDEX_ID=ME_INDEX_RESOURCE_NAME.split("/")[5]


REQUESTS_PER_MINUTE = 200 # project quota==300
vertex_embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)


me = MatchingEngineVectorStore.from_components(
    project_id=PROJECT_ID
    , region=LOCATION
    , gcs_bucket_name=BUCKET_NAME
    , embedding=vertex_embedding
    , index_id=ME_INDEX_ID
    , endpoint_id=ME_INDEX_ENDPOINT_ID
    , k = 10
)


## Create VectorStore Agent tool

vertex_langchain_agent = LangchainAgent()

vectorstore_agent = vertex_langchain_agent.get_vectorstore_agent(
    vectorstore=me
    , vectorstore_name=f"news on {ACTOR_NAME}"
    , vectorstore_description=f"a vectorstore containing news articles and current events for {ACTOR_NAME}."
)

## BigQuery Agent 


vertex_langchain_agent = LangchainAgent()
bq_agent = vertex_langchain_agent.get_bigquery_agent(PROJECT_ID)


bq_agent_tools = bq_agent.tools

bq_agent_tools[0].description = bq_agent_tools[0].description + \
 f"""
  only use the schema {MY_BQ_TRENDS_DATASET}
  NOTE YOU CANNOT DO OPERATIONS AN AN AGGREGATED FIELD UNLESS IT IS IN A CTE WHICH IS ALLOWED
  also - use a like operator for the term field e.g. WHERE term LIKE '%keyword%' 
  make sure to lower case the term in the WHERE clause
  be sure to LIMIT 100 for all queries
  if you don't have a LIMIT 100, there will be problems
 """


## Build an Agent that has access to Multiple Tools

llm = VertexAI()

dataset = 'google_trends_my_project'

db = SQLDatabase.from_uri(f"bigquery://{PROJECT_ID}/{dataset}")

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

me_tools = vectorstore_agent.tools

search = DuckDuckGoSearchAPIWrapper()

    
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
]


# tools.extend(me_tools) #TODO - this is not working on a local macbook; may work on cloudtop or other config
tools.extend(bq_agent_tools)

# Run the streamlit app

# what are the unique terms in the top_rising_terms table?

enable_custom = True
# Initialize agent
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with st.form(key="form"):
    user_input = ""
    user_input = st.text_input("Ask your question")
    submit_clicked = st.form_submit_button("Submit Question")

    
output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)
    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)
    answer = mrkl.run(user_input, callbacks=[st_callback])
    answer_container.write(answer)


"#### Here's some info on the tools in this agent: "
for t in tools:
    st.write(t.name)
    st.write(t.description)
    st.write('\n')

