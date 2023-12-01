from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, GPTListIndex
from langchain import OpenAI
import gradio as gr
import sys
import os

# pip install langchain==0.0.118
# pip install gpt_index==0.4.24
#
# please create folders for source data like: dataSet1, dataSet2 ...
# Then change the value of numOfFolder
#
numOfFolder = 2

def construct_index(directory_path, index):
    print(directory_path)
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index'+str(index)+'.json')
    return index

def construct_index(directory_path, folderIndex):
    print(directory_path)
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('result/index'+str(folderIndex)+'.json')
    return index

def chatbot(input_text):
    indexes = []
    for x in range(numOfFolder):
      y = x + 1;
      index1 = GPTSimpleVectorIndex.load_from_disk('result/index'+ str(y)+'.json')
      index1.set_text("summary" + str(y))
      indexes.append(index1)
    index = GPTListIndex(indexes)
    response = index.query(input_text, response_mode="compact")
    return response.response

os.environ["OPENAI_API_KEY"] = 'your open API KEY'

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Ask your financial questions"),
                     outputs="text",
                     title="FinGPT - Your personal financial advisor")

for x in range(numOfFolder):
  folderInex = x + 1;
  index = construct_index("dataSet" + str(folderInex), folderInex)

iface.launch(server_name="0.0.0.0")

