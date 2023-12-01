#%%
#region Libraries
import pandas as pd
import streamlit as st
from typing import List
from langchain.llms import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.chains import SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
#endregion

#region Call Models and Prepare Data
llm = VertexAI(model_name= "text-bison-32k" , max_output_tokens=8192, temperature=0.4)
embeddings = VertexAIEmbeddings()
df = pd.read_csv("faqsamples.csv")
data = df.iloc[:,1:]

context = ""
for index, row in data.iterrows():
    context = context + "question: " + row.Question + " answer: " + row.Answer + "|" + "\n"
#endregion

#region New chain component
# Create the output schema to give llm response structure (models always send outputs as string [str])
response_schemas = [
    ResponseSchema(name="intent", description="the intent for the question answer pair."),
    ResponseSchema(name="question", description="the question from the data context."),
    ResponseSchema(name="answer", description="the answer from the data context.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Template to create the prompt
prompt_template1 = PromptTemplate(
    input_variables = ["context"],
    template = """
        Extract the intent from the following data(context) per each row separated by pipe |: 
        
        {context}
                        
        {format_instructions}
    """,
    partial_variables={"format_instructions":  format_instructions}
)

# Component of langchain with the prompt to chain
chain_1 = LLMChain(
    llm = llm,
    prompt = prompt_template1,
    output_key = "detected_intent"
)

# Create the output schema to give llm response structure (models always send outputs as string [str])
class ResponseItem(BaseModel):
    new_intent: str = Field(description="the intent for the question answer pair")
    new_questions: List[str] = Field(description="the list of all the new questions tied to the intent")
    answer: List[str] = Field(description="the list of the answer to your new questions")
class Response(BaseModel):
    new_intent: List[ResponseItem]

parser = PydanticOutputParser(pydantic_object=Response)

# Template to create the prompt
prompt_template2 = PromptTemplate(
    input_variables = ["detected_intent"],
    template = """
        Your task is to generate new different additional questions a user might have from the answers below in the data/context:
        
        data: {detected_intent}
        
        - Do not copy or repeat questions, use your creativity to desing new ones.
        
        {format_instructions}           
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Component of langchain with the prompt to chain
chain_2 = LLMChain(
    llm = llm,
    prompt = prompt_template2,
    output_key = "new_intent"
    )

# Join of chain together
overall_chain = SequentialChain(
    chains=[chain_1, chain_2],
    input_variables=["context"],
    # Here we return multiple variables
    output_variables=["new_intent"],
    verbose=True)

res = overall_chain.run(context)
print(res)
response = parser.parse(res)
print(response)
#endregion
# %%

#region Creating a CSV file
import csv
with open("faq_new_q.csv", "w") as f:
    #writer = csv.writer(f, delimiter=",", lineterminator="\n")
    data_f_res = []
    for i in response.new_intent:
        for num, new in enumerate(i.new_questions):
            new_data = "'" + i.new_intent + "'" + "," + "'" + str(new) + "'" + "," + "'" + str(i.answer[num]) + "'"
            f.write(new_data + "\n")
#endregion
