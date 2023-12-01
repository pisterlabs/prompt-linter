from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
import os
import emoji
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
# import gradio as gr
import sys
import os

from llama_index import ServiceContext


# NOTE: for local testing only, do NOT deploy with your key hardcoded
os.environ['OPENAI_API_KEY'] = "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"

def embeddings_to_chats(prompt, inputfolder, debug=False):
    
    if False and os.path.exists("gacetachat.json"):
        # load from disk
        index = GPTSimpleVectorIndex.load_from_disk('gacetachat.json')
    else:
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        if debug:
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=1, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
        else:
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=1, model_name="text-davinci-003", max_tokens=num_outputs))

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        documents = SimpleDirectoryReader(inputfolder).load_data()

        # add try and except for the case where the user has exceeded his daily limit
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    response = index.query(prompt)
    
    if not os.path.exists("gacetachat.json"):
        index.save_to_disk('gacetachat.json')

    with open("output.txt", "a", encoding='utf-8') as output_file:
        output_file.write(response.response)

    return response.response
