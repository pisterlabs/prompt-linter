import os
import openai
import dotenv

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

template = """
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.

The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1, 1. 
A:"""

prompt = PromptTemplate(
    input_variables=[],
    template=template
)

llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

print(f"\n{chain.predict()}\n")