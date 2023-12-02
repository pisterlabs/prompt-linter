# %%
import openai
import os
from langchain import OpenAI

from langchain.chains import LLMChain, MapReduceChain
from langchain import PromptTemplate
from langchain import *
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document



# %%
prompt_template = """

Given a summary of a machine learning paper from Arxiv and some existing code, please generate a Python script that implements and replicates the results from the paper EXACTLY WITH NO ERRORS. Start from the provided code and modify it as necessary to incorporate the new ideas from the paper. The script should be organized in a logical and coherent way, with sections and sub-sections that reflect the paper's organization. Assume that the notebook will be used by someone who is familiar with the general field of AI, but may not be familiar with the specific topic of the paper. Please include code, explanations, and visualizations where appropriate. The script should be written in Python using the PyTorch machine learning framework and should be easy to run without any additional setup. Please do not use any external libraries or packages that are not already included in the provided code.

######################
LaTeX Source for paper
######################
    {paper}

######################
Previous Code
######################

    {code}


Now, it's time to continue this code. Please remember to write ORIGINAL code without copying whatever is in the "Previous Code" section. The goal is to write code that is as close as possible to what a human would write. If you copy code from the previous section, the model will learn to copy code instead of writing original code. If you are stuck, you can use the "Previous Code" section as a starting point, but please modify it as much as possible.

######################
Your Code Here
######################
"""

def generate_code(paper_str: str, model_name: str = "code-davinci-002", code="import torch"):

    llm = OpenAI(model_name=model_name)
    code = "import torch"

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["paper", "code"]
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    # chain = MapReduceChain.from_params(llm=llm, prompt=PROMPT, text_splitter=RecursiveCharacterTextSplitter())

    return chain.apply([{"paper": paper_str, "code": code}])[0]['text']




summary_prompt_template = """

Given the LaTeX source of a machine learning paper from Arxiv and some existing code, please generate a summary that can be used to implement and replicate the results from the paper EXACTLY WITH NO ERRORS. The summary should be organized in a logical and coherent way, with sections and sub-sections that reflect the paper's organization. BE SPECIFIC AND INCLUDE ALL DETAILS ABOUT THE MODEL ARCHITECTURE AND TRAINING SETUP. Assume that this summary will be used by someone who is familiar with the general field of AI, but may not be familiar with the specific topic of the paper. Please include code and explanations where appropriate. Be sure to include all the important numbers ande details that a large language model might need to write the code that replicates the paper.

######################
LaTeX Source for paper
######################
    {paper}

#############################################
Summary and instructions for generating code
#############################################
"""



def summarize_paper(paper_str: str, model_name: str = "code-davinci-002"):

    llm = OpenAI(model_name=model_name)
    code = "import torch"

    PROMPT = PromptTemplate(
        template=summary_prompt_template, input_variables=["paper"]
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    # chain = MapReduceChain.from_params(llm=llm, prompt=PROMPT, text_splitter=RecursiveCharacterTextSplitter())

    return chain.apply([{"paper": paper_str}])[0]['text']


    llm = OpenAI(model_name=model_name)