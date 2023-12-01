import os
from dotenv import load_dotenv
# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
from langchain import PromptTemplate

llm = OpenAI(model_name="text-davinci-003", temperature=0.5)

#Number-1: Using simple promt
our_prompt = """"
I love trips, and I have been to 6 countries.
I plan to visit few more soon.

Can you create a post for tweet in 10 words or above?
"""
print("Output 1 :-",llm(our_prompt)) # Prints the model output 

#Number-2: Using f strings
wordsCount = 3
our_text = "I love trips, and I have been to 6 countries. I plan to visit few more soon."
our_prompt = f"""{our_text} 
Can you create a post for a tweet in {wordsCount} words or above ?
"""
print("Output 2 :-",llm(our_prompt)) 

#Number-3: Using prompt Template (It keeps the code more neat and clean while solving complex problems)
template = """"
{our_text}
Can you create a post for a tweet in {wordsCount} words or above ?
"""
prompt = PromptTemplate(
    input_variables = ["wordsCount","our_text"],
    template = template
)

final_prompt = prompt.format(wordsCount='14', our_text="I love trips, and I have been to 6 countries. I plan to visit few more soon.")
print("Output 3 :-",llm(final_prompt))