import os

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
from comet_llm import Span, end_chain, start_chain

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def get_llm_chain() -> LLMChain:

    # prompte template
    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    # llm
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    # llm chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain

llm_chain = get_llm_chain()

def main(question: str):

    start_chain(inputs={"question": question})

    with Span(
        category="llm-reasoning",
        inputs={"question": question},
    ) as span:
        response = llm_chain.run(question)
        print(response)
        span.set_outputs(outputs={"response": response})

    end_chain(outputs={"response": response})

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
main(question)