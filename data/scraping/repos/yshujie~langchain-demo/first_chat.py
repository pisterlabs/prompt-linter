from langchain.llms import OpenAI

def evaluateAI():
    llm = OpenAI(model_name="text-davinci-003", max_tokens=1024)
    print(llm("怎么评价人工智能？"))