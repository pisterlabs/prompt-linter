from gpt_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = "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"


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs
        )
    )
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    index.save_to_disk("index.json")

    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk("index.json")
    response = index.query(input_text, response_mode="compact")
    return response.response


index = construct_index("docs")


iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-trained AI Chatbot",
)
iface.launch(share=True)
