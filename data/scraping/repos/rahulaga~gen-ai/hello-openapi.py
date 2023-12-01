from langchain.llms import OpenAI

llm = OpenAI(verbose=True)
text = "What would be good stocks to buy for a hypothetical high risk portfolio"
print(llm(text))