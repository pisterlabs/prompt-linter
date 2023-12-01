from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
text = "Translate 'Hello world!' to German."
print(llm(text))
