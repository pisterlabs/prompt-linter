import zipfile

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

import langchain
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(
    database_path=".langchain.db"
)  # caches queries that are the same.

def generate_code(question, model_type, api_key):
    """
    Generate Python code script. The script should only include code, no comments.
    What is returned is code that needs some light modification.
    """
    task = "Generate Python Code Script. The script should only include code, no comments."
    chat_model = ChatOpenAI(temperature=0, model=model_type, openai_api_key = api_key)
    response = chat_model(
        messages=[
            SystemMessage(content=task),
            HumanMessage(content=question)
        ],
    )
    code = response.content
    return format_code(code)


def format_code(code):
    """
    Format the generated code to remove read_csv if its there.
    """
    csv_line_index = code.find("read_csv")
    if csv_line_index > 0:
        before_csv_line_index = code[0:csv_line_index].rfind("\n")
        before_csv = (
            "" if before_csv_line_index == -1 else code[0:before_csv_line_index]
        )
        after_csv = code[csv_line_index:]
        after_csv_line_index = after_csv.find("\n")
        after_csv = (
            "" if after_csv_line_index == -1 else after_csv[after_csv_line_index:]
        )
        return before_csv + after_csv
    return code


def format_primer(primer_desc, primer_code, question):
    """
    Format the primer description and code.
    """
    return f'"""\n{primer_desc}{question}\n"""\n{primer_code}'


def generate_primer(df, df_name):
    """
    Generate a primer for visualization. Include all the necessary imports.
    """
    primer_desc = """
    If a graph is needed, label the x and y axes appropriately.
    Add a title. Use st.plotly or plotly.express to visualize."
    Using Python version 3.9.12, only create a script using the dataframe "df" if the user asked for a graph. 
    Do not give a y/n answer in the code or try to take any user input. 
    Graph should be created using st.plotly or plotly.express. If there is only a single row, manipulate it to be able to chart the data.
    IF the given sql query is only one column, manipulate it to get it in the right format for a chart if it has multiple rows.
    If its just one column one row, don't chart, just display the table.
    Make sure that variable names are the same as the sql.
    """
    primer_code = (
        "import pandas as pd\nimport streamlit as st\nimport plotly.express as px\n"
    )
    primer_code += f"df = {df_name}.copy()\n"
    return primer_desc, primer_code

def unzip(file):
    with zipfile.ZipFile(file, 'r') as zip_ref:
      zip_ref.extractall()
