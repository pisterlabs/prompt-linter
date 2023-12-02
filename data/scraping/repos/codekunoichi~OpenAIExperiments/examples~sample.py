import os
import openai
openai.organization = "org-JCW7HsSGdQfwZzo2lZBIQLGM"
openai.api_key = os.getenv("OPENAI_API_KEY")

print("\n List models:")
openai.Model.list()

# list engines
engines = openai.Engine.list()

# print the first engine's id
print("\n print the first engine's id:")
print(engines.data[0].id)

# create a completion
completion = openai.Completion.create(engine="ada", prompt="Hello world")

# print the completion
print("\n print the completion:")
print(completion.choices[0].text)
