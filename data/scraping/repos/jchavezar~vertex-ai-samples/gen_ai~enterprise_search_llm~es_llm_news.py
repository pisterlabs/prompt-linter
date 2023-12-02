#%%
!pip install langchain
!pip install google-cloud-aiplatform
!pip install google-cloud-discoveryengine

#%%
#@title ### You will need to update these values
VERTEX_API_PROJECT = 'vtxdemos' #@param {"type": "string"}
VERTEX_API_LOCATION = 'us-central1' #@param {"type": "string"}

# @title Imports and custom PaLM classes (credit `rthallam@`)
from langchain import PromptTemplate
from langchain.chains import LLMChain
from utils.ai import VertexLLM, EnterpriseSearchChain
    
#@title Initialise the LLM
GCP_PROJECT = "vtxdemos" #@param {type: "string"}
SEARCH_ENGINE = "news_1687453492092" #@param {type: "string"}
LLM_MODEL = "text-bison@001" #@param {type: "string"}
MAX_OUTPUT_TOKENS = 1024 #@param {type: "integer"}
TEMPERATURE = 0.2 #@param {type: "number"}
TOP_P = 0.8 #@param {type: "number"}
TOP_K = 40 #@param {type: "number"}
VERBOSE = True #@param {type: "boolean"}

llm_params = dict(
    model_name=LLM_MODEL,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=VERBOSE,
)

llm = VertexLLM(**llm_params)

##--------------------------------------------------------------------------------------------------------------------------------------------------##
# %%
#@title Example - summarize financial results
SEARCH_QUERY = 'News about Harrison Ford' #@param {type: "string"}
PROMPT_STRING = "Please parse these search results of this subject and combine them into a tab delimited table: {results}, the table should contain, date, summary and reference" #@param {type: "string"}

# Combine the LLM with a prompt to make a simple chain
prompt = PromptTemplate(input_variables=['results'],
                        template=PROMPT_STRING)

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# Combine this chain with Enterprise Search in a new chain
es_chain = EnterpriseSearchChain(project=GCP_PROJECT,
                                 search_engine=SEARCH_ENGINE,
                                 chain=chain)

result = es_chain.run(SEARCH_QUERY)

result.split('\n')
# %%
