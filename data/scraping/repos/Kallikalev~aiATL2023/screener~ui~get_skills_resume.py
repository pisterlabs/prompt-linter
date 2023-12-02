import os
from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain


template = """
You are an technical recruiter screening candidate resumes.
Identify the candidates skills specific to programming and software development, and output in a array list format. Limit skills to programming languages, frameworks, and programming concepts.

Format your output according to the following structure:

['skill 1','skill 2', 'skill 3', ...]

Resume: {resume}
"""
llm = VertexAI(model_name="code-bison",max_output_tokens=1000, temperature=0.1)
prompt = PromptTemplate(template=template,input_variables=["resume"])
llm_chain = LLMChain(prompt=prompt,llm=llm)

def get_skills(resumeText):
    return llm_chain.run(resumeText)