import openai
import sys
sys.path.insert(0, '/Users/shreyagarwal/Code/GitHub/MATH-GPT/pal')
import pal
from pal.prompt import math_prompts
from langchain import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import PALChain
import os

def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

api_key = get_file_contents('api_key.txt')

openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

# MODEL = 'text-davinci-003' #@param {type:"string"}m

# interface = pal.interface.ProgramInterface(model=MODEL, get_answer_expr='solution()', verbose=True)

# question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"#@param {type:"string"}
# prompt = math_prompts.MATH_PROMPT.format(question=question)

# answer = interface.run(prompt, time_out=10)
# print('\n==========================')
# print(f'The answer is {answer}.')

llm = OpenAI(model_name='text-davinci-003', temperature=0, max_tokens=512)
pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"
pal_chain.run(question)