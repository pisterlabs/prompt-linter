from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
os.environ["OPENAI_API_KEY"] = open_file('key_openai.txt')

def construct_index(directory_path):
  # set maximum input size
  max_input_size = 4096
  # set number of output tokens
  num_outputs = 256
  # set maximum chunk overlap
  max_chunk_overlap = 20
  # set chunk size limit
  chunk_size_limit = 600

  prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
  
  documents = SimpleDirectoryReader(directory_path).load_data()
  
  index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  
  index.save_to_disk('index.json')
  
  return index
  
def ask_bot(input_index = 'index.json'):
  index = GPTSimpleVectorIndex.load_from_disk(input_index)
  while True:
    query = input('User: \n')
    response = index.query(query, response_mode="compact")
    print ("\nJJ: " + response.response + "\n")

    
if __name__ == '__main__':
    index = construct_index("data/")
    ask_bot('index.json')

