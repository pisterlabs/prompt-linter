from langchain.chains import LLMCheckerChain
from langchain.models import OpenAI

llm = OpenAI(temperature=0.7)
checker_chain = LLMCheckerChain.from_llm(llm)

text1 = "This is the first text."
text2 = "This is the second text."

output = checker_chain.run(text1, text2)

print(output)