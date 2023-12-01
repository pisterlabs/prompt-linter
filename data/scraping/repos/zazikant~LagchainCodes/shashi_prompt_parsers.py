import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import ConversationSummaryBufferMemory


load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

#import llm and ask question    

#llm = OpenAI(model="text-ada-001")

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-token"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-token"

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"max_length":64, "max_new_tokens":500, "temperature":0.1})


context = """
One of the learnings from our recent projects is that the little things matter. Take for example daily housekeeping. How this helps? 

#1 Getting started with a task takes the longest time. A clean workspace encourages you to hit the Go button! 

# 2 The start to the day is more productive if workers find tools and materials which they need to use. 

# 3 An orderly workspace naturally ensures better safety and helps avoid slipping and tripping. 

# 4 “Small wins” is an unbelievable and under-rated productivity hack! Seeing a clean workfront at the beginning of the day provides a subliminal trigger to the brain that things are going well, pushing us into an upward performance cycle.

Focus on the little things and grab the small wins to see the difference for yourself.
""".strip()


sentiment_schema = ResponseSchema(
    name="sentiment",
    description="Is the text positive, neutral or negative? Only provide these words",
)
subject_schema = ResponseSchema(
    name="subject", description="What subject is the text about? Use 2 lines."
)

response_schemas = [sentiment_schema, subject_schema]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
format_instructions


template = """
Interprete the text and evaluate the text.
sentiment: Is the text positive, neutral or negative? Only provide these words.
subject: What subject is the text about? Use 2 lines

text: {context}

Just return the JSON, do not add ANYTHING, NO INTERPRETATION!
{format_instructions}:"""

#imprtant to have the format instructions in the template represented as {format_instructions}:"""

#very important to note that the format instructions is the json format that consists of the output key and value pair. It could be multiple key value pairs. All the context with input variables should be written above that in the template.

prompt  = PromptTemplate(
    input_variables=["context", "format_instructions"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt, output_key= "testi")
response = chain.run({"context": context, "format_instructions": format_instructions})
response

output_dict = parser.parse(response)
output_dict

