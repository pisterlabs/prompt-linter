from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI

max_input_size = 3700
num_outputs = 300
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
prompt_helper = PromptHelper.from_llm_predictor(llm_predictor)

def construct_index(directory_path, index_name):
  documents = SimpleDirectoryReader(directory_path, recursive=True).load_data()
  index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  index.save_to_disk(index_name)
  return index

construct_index('./training-data', 'index_new.json')