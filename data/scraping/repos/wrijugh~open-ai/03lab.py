from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

import os
import openai
from dotenv import load_dotenv

if load_dotenv("../.env"):
    print("Found OpenAPI Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
else: 
    print("No file .env found")

# Create an instance of Azure OpenAI
llm = AzureChatOpenAI(
    openai_api_type = openai.api_type,
    openai_api_version = os.getenv("OPENAI_API_VERSION"),
    openai_api_base = os.getenv("OPENAI_API_BASE"),
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
)

# Define the prompt we want the AI to respond to - the message the Human user is asking
msg = HumanMessage(content="Explain step by step. How old is the president of USA?")

# Call the API
r = llm(messages=[msg])

# Print the response
print(r.content)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create a prompt template with variables, note the curly braces
prompt = PromptTemplate(
    input_variables=["input"],
    template="What interesting things can I make with a {input}?",
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
response = chain.run({"input": "raspberry pi"})

# As we are using a single input variable, you could also run the string like this:
# response = chain.run("raspberry pi")

print(response)