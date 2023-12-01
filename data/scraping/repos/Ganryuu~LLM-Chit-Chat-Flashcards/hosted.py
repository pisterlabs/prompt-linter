import os
from langchain.llms import CerebriumAI
from langchain import PromptTemplate, LLMChain


os.environ["CEREBRIUMAI_API_KEY"] = "public-"

template = """
You are a friendly chatbot assistant that responds in a conversational
manner to users questions. Keep the answers short, unless specifically
asked by the user to elaborate on something.
Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = CerebriumAI(
  endpoint_url="https://run.cerebrium.ai/gpt4-all-webhook/predict",
  max_length=100
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

green = "\033[0;32m"
white = "\033[0;39m"

while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        break
    if query == '':
        continue
    response = llm_chain(query)
    print(f"{white}Answer: " + response['text'])