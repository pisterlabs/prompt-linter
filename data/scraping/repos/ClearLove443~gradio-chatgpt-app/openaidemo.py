import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pandasai import PandasAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


def to_str(res):
    type_res = type(res)
    test = str(res)
    print(test)
    if type_res is str:
        return res
    if type_res is str:
        return res


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
            "China",
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
            14631844184064,
        ],
        "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12],
    }
)

# Instantiate a LLM
from pandasai.llm.openai import OpenAI

llm = OpenAI(api_token=API_KEY)

pandas_ai = PandasAI(llm, save_charts=True)
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = pandas_ai(
        df,
        prompt="Plot the histogram of countries showing for each the gdp, using different colors for each bar?",
    )
    print(result)
    print(cb)
