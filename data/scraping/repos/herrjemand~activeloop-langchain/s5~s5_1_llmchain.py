from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain import PromptTemplate, OpenAI, LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("What is the word to replace the following: {word}")
)

print(llm_chain("artificial"))

input_list = [
    {"word": "intelligence"},
    {"word": "learning"},
    {"word": "robot"},
]

print(llm_chain.apply(input_list))

print(llm_chain.generate(input_list))


# Predict

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Looking at the context of '{context}', What is an appropriate word to replace the following: {word}",
        input_variables=["context", "word"],
    ),
)

print(llm_chain.predict(word="fan", context="objects"))
print(llm_chain.predict(word="fan", context="humans"))