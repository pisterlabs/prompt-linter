from llama_index import VectorStoreIndex, GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
import openai
import os

openai_api_key =

WikipediaReader = download_loader('WikipediaReader')

loader = WikipediaReader()
dream_wiki = loader.load_data(pages=['Dream Interpretation'])

dream_wiki_index = GPTSimpleVectorIndex(dream_wiki)
response = dream_wiki_index.query('What is dream interpretation?')
print(response)
