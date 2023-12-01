import os
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor
from langchain.chat_models import ChatOpenAI

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

def tune_llm(input_directory="sourcedata", output_file="indexdata/index.json"):
    loaded_content = SimpleDirectoryReader(input_directory).load_data()

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'))
    output_index = GPTSimpleVectorIndex(loaded_content, llm_predictor=llm_predictor)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output_index.save_to_disk(output_file)