import os
import openai
import dotenv

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

template = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

....

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5

Q: When I was 6 my sister was half my age. Now I’m 70 how old is my sister?
A:
"""

prompt = PromptTemplate(
    input_variables=[],
    template=template
)

llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

for i in range(5):
  print(f"Output {i+1}\n {chain.predict()}")