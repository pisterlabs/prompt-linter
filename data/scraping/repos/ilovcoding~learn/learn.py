import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI()

promptZhCN = PromptTemplate(
    input_variables=["product"],
    template="怎么做可以精通 {product}",
)

print(promptZhCN.format(product="React"))


print("\n****************************\n")
zhCN_Chain = LLMChain(llm=llm, prompt=promptZhCN)
print(zhCN_Chain.run("React"))
print(zhCN_Chain.run("Vue"))
print(zhCN_Chain.run("Angular"))

