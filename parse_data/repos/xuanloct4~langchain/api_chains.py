import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate

# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0)

from langchain.chains.api import open_meteo_docs
chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)
chain_new.run('What is the weather like right now in Munich, Germany in degrees Farenheit?')

##TMDB API
import os
from langchain.chains.api import tmdb_docs
headers = {"Authorization": f"Bearer {os.environ['TMDB_BEARER_TOKEN']}"}
chain = APIChain.from_llm_and_api_docs(llm, tmdb_docs.TMDB_DOCS, headers=headers, verbose=True)
chain.run("Search for 'Avatar'")


from langchain.chains.api import podcast_docs
from langchain.chains import APIChain

##listennotes API
# Get api key here: https://www.listennotes.com/api/pricing/
# os.environ['LISTENNOTES_API_KEY'] = 'xxx'

headers = {"X-ListenAPI-Key": os.environ['LISTENNOTES_API_KEY']}
chain = APIChain.from_llm_and_api_docs(llm, podcast_docs.PODCAST_DOCS, headers=headers, verbose=True)
chain.run("Search for 'silicon valley bank' podcast episodes, audio length is more than 30 minutes, return only 1 results")