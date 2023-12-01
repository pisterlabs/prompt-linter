from langchain.llms import OpenAI
from langchain.chains import LLMChain, LLMSummarizationCheckerChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, SequentialChain
from pymongo import MongoClient
import os
from dotenv import load_dotenv

def fact_check(answer):
    llm = OpenAI(temperature=0)

    dict = {"answer": answer}
    template = """Here is the answer: {answer} to the question. Make a bullet point list of the assumptions in the answer.\n\n"""
    prompt_template = PromptTemplate(input_variables=["answer"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="assertions")

    template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="facts")

    template = """In light of the above facts, how would you answer the question and explain your answer."""
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="output")

    overall_chain = SequentialChain(chains=[assumptions_chain, fact_checker_chain, answer_chain],
                                    input_variables=["answer"],
                                    output_variables=["assertions", "facts", "output"],
                                    verbose=True)

    return overall_chain(dict)


def checker_chain(answer):
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    db = client["answer_checks"]
    col = db["checks"]
    llm = OpenAI(temperature=0)
    checker_chain = LLMSummarizationCheckerChain.from_llm(llm, max_checks=3, verbose=True)
    checks = checker_chain.run(answer)
    dict = {"answer": answer, "checks": checks}
    col.insert_one(dict)
    print(checks)
        