import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


# Define tagging function
class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    symptom: str = Field(description="main symptom of the patient, should be `pain`, `fatigue`, `flu symptoms` or `sleep problems`")
    severity: str = Field(description="severity of symptom, should be `mild`, `moderate`, or `severe`")

convert_pydantic_to_openai_function(Tagging)


# define OpenAImodel, functions and prompt
model = ChatOpenAI(temperature=0)

tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])


# chain together!
tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
print(tagging_chain.invoke({"input":"I am not feeling well, I am coughing a lot, and has a bit headache."}))