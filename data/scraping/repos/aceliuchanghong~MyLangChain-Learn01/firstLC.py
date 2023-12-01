from langchain.llms import OpenAI


llm = OpenAI(temperature=0.9)

#注意,代理设置全局代理

print(llm("What would be a good company name for a company that makes colorful socks?"))
print(llm("怎么评价人工智能"))

print("END")