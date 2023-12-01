# PAL: Program-aided Language Models.
# DEPRECATED ???

import os
import openai
import dotenv


from langchain.chains import PALChain
from langchain import OpenAI

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

llm = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=512)
pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"

pal_chain.run(question)