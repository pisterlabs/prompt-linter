from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from templates import template_string, generate_string
from io import StringIO
import openai
import os 


# Load API key
config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# Use the OpenAI LLM
chat = ChatOpenAI(temperature=0.0)

#Create a langchain template 
generate_template = ChatPromptTemplate.from_template(generate_string)
#Input any issue for the developers to about
issue = input("Enter any topic for the transcript")
print("Generating the transcript")
transcript_prompt = generate_template.format_messages(issue=issue)
print(transcript_prompt)
transcript = chat(transcript_prompt)
print(transcript)
# transcript = chat(generate_string)
# csv_file = StringIO(transcript)
# print(csv_file)

