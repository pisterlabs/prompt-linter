import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from .prompt_control import extract_variables_with_replacement
import re
import json

import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

### functions to display information 
def show_prompts_and_variables(prompt_text:str):
    # find occruences of 2 or more \n and insert __ before and after
    prompt_text = re.sub(r"\n{2,}", r"__\n\n__", prompt_text)
    st.markdown(f"__{prompt_text}__")
    matches, _ = extract_variables_with_replacement(prompt_text)
    st.caption(f"Variables detected in the prompts:  {', '.join(map(lambda x: f':red[*{x}*]',matches))}.") if matches else st.caption("No variables detected in the prompts.")
    if st.button("Modify the prompt.", key=prompt_text[:10]):
        switch_page("Prompts")
    return matches

def show_response_schemas(response_schemas:list):
    rs = pd.DataFrame(response_schemas)
    st.dataframe(rs)
    return rs

def get_response_schema(rs_df):
    return [
        ResponseSchema(
            name=schema[0], description=schema[1],
        ) 
        for schema in rs_df.values.tolist()
    ]

def parse_md_code(ans:str):
    code = ans[ans.find("```") + 3:]
    code = code[:code.find("```")]
    if code.index('"code"') > -1:
        code = code[code.index('"code"') + 6:]
        code = code[code.find('"')+1:]
        code = code[:code.find('"')]
    return code

def get_prompt(rs_df:pd.DataFrame, template:str, variables:dict):
    # schema response 
    response_schemas = get_response_schema(rs_df)
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    args = {k:"\n".join(v) if isinstance(v, list) else v for k,v in variables.items()}
    template = "{template}\n{{response_schemas}}".format(template=template)
    prompt = PromptTemplate(
        template=template,
        input_variables=[*args],
        partial_variables={"response_schemas":format_instructions}
    )
    return prompt.format(**args)