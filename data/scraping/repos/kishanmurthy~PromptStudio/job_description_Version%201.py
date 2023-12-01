import sys
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

apikey = sys.argv[1]

def generate_openai_output(prompt, input_dict, input_tags, apikey):
	prompt_tempt = PromptTemplate.from_template(prompt)
	filtered_dict = {key: input_dict[key] for key in input_tags}
	final_prompt = prompt_tempt.format(**filtered_dict)
	llm = OpenAI(openai_api_key=apikey)
	output = llm.predict(final_prompt)
	return output

a= ""

input_dict = {}
prompt = "tell me about {a}"
input_dict["a"] = a
input_tags = ["a"]
o= generate_openai_output(prompt, input_dict, input_tags, apikey)

print(o)
