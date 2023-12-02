from langchain import PromptTemplate

simple_template = "This is my template."

prompt = PromptTemplate(
  input_variables=[],
  template=simple_template,
)

formatted_prompt = prompt.format()

print(formatted_prompt)