import openai
import langchain

import os
os.environ["OPENAI_API_KEY"] = "sk-tkZioLUcopU4zLZFxHPCT3BlbkFJBFxWHxNdm5CZZ9vMYIg"


llm = openai(temperature=0.9)
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))