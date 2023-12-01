import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# df = pd.DataFrame({
#     "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
#     "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
#     "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
# })

df = pd.DataFrame(
    {
        "country": [
            "United States",
            "United Kingdom",
            "France",
            "Germany",
            "Italy",
            "Spain",
            "Canada",
            "Australia",
            "Japan",
        ],
        "gdp": [
            19294482071552,
            2891615567872,
            2411255037952,
            3435817336832,
            1745433788416,
            1181205135360,
            1607402389504,
            1490967855104,
            4380756541440,
        ],
        "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87],
    }
)

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm, verbose=True)


async def make_completion(history):
    import matplotlib as mpl

    mpl.use("TkAgg")
    # Instantiate a LLM
    from langchain.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        res = pandas_ai.run(df, prompt=history[-1])
        print(cb)
        return str(res)
        if isinstance(res, pd.Series):
            return res.to_string()
        else:
            return str(res)


async def predict(input, history):
    """
    Predict the response of the chatbot and complete a running list of chat history.
    """
    history.append(input)
    response = await make_completion(history)
    history.append(response)
    messages = [(history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)]
    return messages, history


css = """
.contain {margin-top: 80px;}
.title > div {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo:
    logger.info("Starting Demo...")
    chatbot = gr.Chatbot(label="Chatbot", elem_classes="title")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(
            show_label=False, placeholder="Enter text and press enter"
        ).style(container=False)

    txt.submit(predict, [txt, state], [chatbot, state])


demo.launch(server_port=8080, share=True, server_name="0.0.0.0")
