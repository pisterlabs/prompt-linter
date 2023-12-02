# Import required libraries and modules
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Load environment variables (e.g. API keys)
load_dotenv()

# Define the prompt template to instruct the language model
# This template is more explicit in what is expected in the response
template = """
Question: {question}
Please provide three possible names along with short backstories for a science fiction RPG character.
Answer:
"""
# Create a prompt template with input variable as 'question'
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize the OpenAI language model with specified settings
llm = OpenAI(model_name="text-davinci-003", temperature=.95)

# Create an LLMChain with the prompt and language model
chain = LLMChain(llm=llm, prompt=prompt)

# Define the question to be passed to the language model
question = "Can you make my character a female programmer from the future with super powers?"

# Run the chain and get the output from the language model
try:
    output = chain.run(question)
    # Print the output
    print(output)
except Exception as e:
    # Print error message if something goes wrong
    print(f"An error occurred: {e}")
