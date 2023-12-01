import os

import gradio as gr
from langchain import VectorDBQA
from langchain.llms import OpenAI

import hpqa

os.environ["OPENAI_API_KEY"] = ""
index_path = hpqa.LOCAL_DATA_PATH / "hpqa_faiss_index_500"


examples = [
    "How would you sneak into Hogwarts without being detected?",
    "Why did Snape kill Dumbledore?",
    "Who is the most badass wizard in the world?",
    "Who would win a fight between Dumbledore and a grizzly bear?",
    "How many siblings does Hermione have?",
    "Why are the Dursleys so mean to Harry?",
]


def api(question, temperature, api_key=None):
    if api_key is None or len(api_key) == 0:
        return "You must provide an OpenAI API key to use this demo 👇"
    if len(question) == 0:
        return ""
    document_store = hpqa.load_document_store(index_path, openai_api_key=api_key)
    chain = VectorDBQA.from_chain_type(
        llm=OpenAI(temperature=temperature, openai_api_key=api_key),
        chain_type="stuff",
        vectorstore=document_store,
        return_source_documents=True,
    )
    response = chain(question)
    return response["result"].strip()


demo = gr.Blocks()

with demo:
    gr.Markdown("# 🪄 The GPT Who Lived: Harry Potter QA with GPT 🤖")
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(lines=4, label="Question")
            temperature = gr.Slider(0.0, 2.0, 0.7, step=0.1, label="🍺 Butterbeer Consumed")
            with gr.Row():
                clear = gr.Button("Clear")
                btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            answer = gr.Textbox(lines=4, label="Answer")
            openai_api_key = gr.Textbox(type="password", label="OpenAI API key")
    btn.click(api, [question, temperature, openai_api_key], answer)
    clear.click(lambda _: "", question, question)
    gr.Examples(examples, question)
    gr.Markdown("💻 Checkout the `hpqa` source code on [GitHub](https://github.com/johnnygreco/hpqa).")
demo.launch()
