from langchain.llms import OpenAI

llm = OpenAI()
print(llm.predict("hi!"))
