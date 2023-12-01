"""
Optional: Change where pretrained models from huggingface will be downloaded (cached) to:
export TRANSFORMERS_CACHE=/whatever/path/you/want
"""

import os
os.environ["TRANSFORMERS_CACHE"] = "./models/transformers_cache"

import time
from langchain.llms.base import LLM
from langchain import OpenAI
from llama_index import (
    GPTListIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Document
)
from transformers import pipeline
import gradio as gr

def timeit():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [str(arg) for arg in args]

            print(f"[{(end - start):.8f} seconds]")
            return result

        return wrapper

    return decorator

prompt_helper = PromptHelper(
    # maximum input size
    max_input_size=2048,
    # number of output tokens
    num_output=1024,
    # the maximum overlap between chunks.
    max_chunk_overlap=20,
)


class LocalOPT(LLM):
    # model_name = "facebook/opt-iml-max-30b" # (this is a 60gb model)
    model_name = "facebook/opt-iml-1.3b"  # ~2.63gb model
    pipeline = pipeline("text-generation", model=model_name)

    def _call(self, prompt: str, stop=None) -> str:
        response = self.pipeline(prompt, max_new_tokens=1024)[0]["generated_text"]
        # only return newly generated tokens
        return response[len(prompt) :]

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"

@timeit()
def build_chat_bot(input):
    global index
    print(input)
    text_list = [input]
    documents = [Document(t) for t in text_list]
    # documents = 
    llm = LLMPredictor(llm=LocalOPT())
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm, prompt_helper=prompt_helper
    )
    index = GPTListIndex.from_documents(documents, service_context=service_context)
    print("Finished indexing")
    
    filename = "storage"
    print("Indexing documents...")
    index.storage_context.persist(persist_dir=f"./{filename}")
    storage_context = StorageContext.from_defaults(persist_dir=f"./{filename}")
    service_context = ServiceContext.from_defaults(llm_predictor=llm, prompt_helper=prompt_helper)
    index = load_index_from_storage(storage_context, service_context = service_context)
    print("Indexing complete")
    return('Index saved')

def chat(chat_history, user_input):
    print("Querying input...")
    query_engine = index.as_query_engine()
    print("Generating response...")
    bot_response = query_engine.query(user_input)

    response_stream = ""
    for letter in ''.join(bot_response.response):
        response_stream += letter + ""
        yield chat_history + [(user_input, response_stream)]
    
    print("Completed response generation")

with gr.Blocks() as demo:
    gr.Markdown('Q&A Bot with Locally Hosted Hugging Face Model')
    with gr.Tab("Input Text Document"):
        # text_input = gr.UploadButton()
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Build")
        text_button.click(build_chat_bot, text_input, text_output)
    with gr.Tab(f"Chatbot"):
        chatbot = gr.Chatbot()
        message = gr.Textbox()
        message.submit(chat, [chatbot, message], chatbot)

demo.queue().launch()

