# Load environment
import os
import pandas as pd
from langchain import PromptTemplate
from pandasai import PandasAI
from pandasai.llm.falcon import Falcon
from langchain.chains import LLMChain
from langchain import HuggingFaceHub
import dotenv
from langchain.llms import OpenAI
import openai

# Load environment variables from .env file
dotenv.load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Load LLM model
#llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"max_length":500, "max_new_tokens":500, "temperature":0.6})

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]

llm = OpenAI(model_name="text-ada-001")

# Load data with pd.read_csv and use pandasai to create a new column named "status"
df = pd.read_csv("Search.csv")
df.shape

llmt = Falcon(HUGGINGFACEHUB_API_TOKEN)
# pandas_ai = PandasAI(llmt, verbose=True, conversational=True)
# response = pandas_ai.run(df, prompt="create a new column and name it as business_search_term. remove the rows having value as Excluded in column Added_or_Excluded")

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

business_schema = ResponseSchema(
    name="Is_Business_Enquiry_keyword",
    description="Answer as Yes or No If the search term relates to seeking service offering of construction pmc in relevance to keyword.",
)

response_schemas = [business_schema]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
format_instructions

template = """
Interprete the search term and keyword and evaluate the text.

Search_Term: {Search_term}
Keyword: {Keyword}

Just return the JSON, do not add ANYTHING, NO INTERPRETATION!
{format_instructions}:"""

#imprtant to have the format instructions in the template represented as {format_instructions}:"""

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

#format_instructions = parser.get_format_instructions()

df

responses = []
# Step 3: Iterate over the DataFrame and generate responses
for index, row in df.iterrows():
    
    row_values = row.to_dict()
    
    prompt = ChatPromptTemplate.from_template(template=template)
    
    chat = ChatOpenAI(temperature=0.0)
    
    input_variables = {"Search_term": row_values["Search_term"], "Keyword": row_values["Keyword"], "format_instructions": format_instructions}
    
    messages = prompt.format_messages(**input_variables)
    
    response = chat(messages)
    
    output_dict = parser.parse(response.content)
    
    # Append response to list
    responses.append(output_dict)
    

# Create new dataframe from responses list
business_term = pd.DataFrame(responses, columns=["Is_Business_Enquiry_keyword"])

# Concatenate the original dataframe with the new dataframe
df = pd.concat([df, business_term], axis=1)

# Export dataframe to CSV
df.to_csv("business_term1.csv", index=False)

df = pd.read_csv("business_term1.csv")

keep_row = 'Yes'

df = df[df['Is_Business_Enquiry_keyword'] == keep_row]

Ads_schema = ResponseSchema(
    name="Ad_copy",
    description="In few lines write google ad copy relating refering to search term that is seeking service offering of construction pmc in relevance to keyword.",
)

response_schemas = [Ads_schema]

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
format_instructions

template = """
You are a helpful assistant in writing ad copies.

Search_Term: {Search_term}
Keyword: {Keyword}

Just return the JSON, do not add ANYTHING, NO INTERPRETATION!
{format_instructions}:"""

responses = []
# Step 3: Iterate over the DataFrame and generate responses
for index, row in df.iterrows():
    
    row_values = row.to_dict()
    
    prompt = ChatPromptTemplate.from_template(template=template)
    
    chat = ChatOpenAI(temperature=0.0)
    
    input_variables = {"Search_term": row_values["Search_term"], "Keyword": row_values["Keyword"], "format_instructions": format_instructions}
    
    messages = prompt.format_messages(**input_variables)
    
    response = chat(messages)
    
    output_dict = parser.parse(response.content)

    # Append response to list
    responses.append(output_dict)
    
# Create new dataframe from responses list

#very important:"Ensure that you are changing New_data to a new name everytime you add a new column"
New_data = pd.DataFrame(responses, columns=["Ad_copy"])

# Concatenate the original dataframe with the new dataframe
df = pd.concat([df, New_data], axis=1)

# Export dataframe to CSV
df.to_csv("business_term2.csv", index=False)