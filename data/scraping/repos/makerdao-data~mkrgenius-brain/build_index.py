from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def construct_index(directory_path, index_name):

  max_input_size = 2048
  num_outputs = 2048
  documents = SimpleDirectoryReader(directory_path, recursive=True).load_data()
	# define LLM
  llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo", max_input_size=max_input_size, num_outputs=num_outputs))
  prompt_helper = PromptHelper.from_llm_predictor(llm_predictor)
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
  index.save_to_disk(index_name)
  
  return index

construct_index('./datasets', 'index_new.json')