from langchain.prompts import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI


llm = ChatOpenAI()

# Basic Example of a Prompt.
response = llm.predict("What are the 7 wonders of the world?")
print(f"Response: {response}")

# Basic Example of a Prompt Template.
prompt_template = PromptTemplate.from_template(
    "List {n} cooking recipe ideas {cuisine} cuisine (name only)."
)
prompt = prompt_template.format(n=3, cuisine="italian")
print(f"Templated Prompt: {prompt}")

response = llm.predict(prompt)
print(f"Response: {response}")
