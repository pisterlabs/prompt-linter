from langchain import PromptTemplate

from console_input import return_for_console_input
from data import TEST_PROMPT_FULL, TEST_PROMPT_HUMAN
from prompt.post_analysis import PostAnalysis
from api.openai import OpenAIAPI
import os

from prompt.prompt_generator import PromptGenerator

print("\n\n\n\n\n")
prompt = input("What topic do you want to mention: ")
res = return_for_console_input(prompt)
print(res)

"""
generator = PromptGenerator()
analysis = PostAnalysis()
api  = OpenAIAPI()
res = api.basic_prompt_response(TEST_PROMPT_FULL)
print(res)
print(1000)
#res = generator.generate_brand_context_response(input_parameters=None, human_template="Write me a post about buying a big mac")
#print(res)

#res = generator.generate_brand_context_response(human_template=TEST_PROMPT_HUMAN, input_parameters=None)
#print(res)
res = api.chat_prompt_response(system_template="You are a helpful assistant that translates {input_language} to {output_language}.",
                               human_template="{text}",
                               input_language="English", output_language="French", text="I love programming.")



#res = analysis.extract_topics()
#print(res)

"""
