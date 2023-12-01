#
#region Install Packages
!pip install langchain
!pip install google-cloud-aiplatform
!pip install google-cloud-discoveryengine
#endregion

#%%
#region Libraries
from langchain import PromptTemplate
from langchain.chains import LLMChain
from utils.ai import VertexLLM, EnterpriseSearchChain
#endregion

#region Variables
VERTEX_API_PROJECT = 'vtxdemos' #@param {"type": "string"}
VERTEX_API_LOCATION = 'us-central1' #@param {"type": "string"}
GCP_PROJECT = "vtxdemos" #@param {type: "string"}
SEARCH_ENGINE = "yext_1691552567184" #@param {type: "string"}
LLM_MODEL = "text-bison@001" #@param {type: "string"}
MAX_OUTPUT_TOKENS = 1024 #@param {type: "integer"}
TEMPERATURE = 0.2 #@param {type: "number"}
TOP_P = 0.8 #@param {type: "number"}
TOP_K = 40 #@param {type: "number"}
VERBOSE = True #@param {type: "boolean"}
#endregion

#region LLM Initialize
llm_params = dict(
    model_name=LLM_MODEL,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=VERBOSE,
)
llm = VertexLLM(**llm_params)
#endregion

# %%
#region Query LLM + ES
query = "a query to create an account entity"
SEARCH_QUERY = 'entities' #@param {type: "string"}
PROMPT_STRING = '''Your task is to help users to create a request url with query parameters using the query search results: {results}
                
                input: a query to list all the account entities my account_id is 9999
                output: curl -X GET https://api.yextapis.com/v2/accounts/999/entities

                input: a query to create an account entity
                output: '''

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
#endregion

# %%
