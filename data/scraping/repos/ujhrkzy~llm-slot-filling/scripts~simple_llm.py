from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

text = "NFTの使用用途としてどんなものが考えられますか。"

prediction = llm(text)
print(prediction.strip())
