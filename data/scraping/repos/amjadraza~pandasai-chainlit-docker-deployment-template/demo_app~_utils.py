import re
import os
from io import BytesIO
from typing import Any, Dict, List
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.llm.open_assistant import OpenAssistant
from pandasai.llm.starcoder import Starcoder
import pandas as pd
import chainlit as cl

models = {
    "OpenAI": OpenAI,
    "Starcoder": Starcoder,
    "Open-Assistant": OpenAssistant
}

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
    "json": pd.read_json,
    "html": pd.read_html,
    "sql": pd.read_sql,
    "feather": pd.read_feather,
    "parquet": pd.read_parquet,
    "dta": pd.read_stata,
    "sas7bdat": pd.read_sas,
    "h5": pd.read_hdf,
    "hdf5": pd.read_hdf,
    "pkl": pd.read_pickle,
    "pickle": pd.read_pickle,
    "gbq": pd.read_gbq,
    "orc": pd.read_orc,
    "xpt": pd.read_sas,
    "sav": pd.read_spss,
    "gz": pd.read_csv,
    "zip": pd.read_csv,
    "bz2": pd.read_csv,
    "xz": pd.read_csv,
    "txt": pd.read_csv,
    "xml": pd.read_xml,
}


def generate_pandasai_response(df,
                               prompt,
                               model_option="OpenAI",
                               is_conversational_answer=False,
                               is_verbose=False):
    """
    A function to run the Query on given Pandas Dataframe
    Args:

        df: A Pandas dataframe
        prompt: Query / Prompt related to data
        model_option: Select the Model from ["OpenAI", "Starcoder", "Open-Assistant"]
        is_conversational_answer: Run model in Conversational mode
        verbose: A parameter to show the intermediate python code generation

    Returns: Response / Results

    """

    user_env = cl.user_session.get("env")
    # os.environ["OPENAI_API_KEY"] = user_env.get("OPENAI_API_KEY")

    llm = models[model_option](api_token=user_env.get("OPENAI_API_KEY"))
    pandas_ai = PandasAI(llm, conversational=False, verbose=is_verbose)
    response = pandas_ai.run(df, prompt=prompt,
                             is_conversational_answer=is_conversational_answer)

    return response