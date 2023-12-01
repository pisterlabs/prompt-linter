import os
import openai
from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")


llm = AzureOpenAI(deployment_name="text-davinci-003", model_name="text-davinci-003")
# Run the LLM
# print(llm("What is the capital of Italy?"))


template = """
I need your expertise as a marketing consultant for a new product launch.

Here are some examples of successful product names:

wearable fitness tracker, Fitbit
premium headphones, Beats
ride-sharing app, Uber
The name should be unique, memorable and relevant to the product.

What is a good name for a {product_type} that offers {benefit}?
"""

prompt = PromptTemplate(
input_variables=["product_type", "benefit"],
template=template,
)

print(llm(
    prompt.format(
        product_type="pair of sunglasses",
        benefit = 'high altitude protection'
    )
))


# with the LLMChain method we were able to combine the two components we needed: the LLM and the prompt template.
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run(product_type="pair of sunglasses", benefit = 'high altitude protection'))

