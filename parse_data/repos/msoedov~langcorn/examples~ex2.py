import os

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-********")


# This is an LLMChain to write a synopsis given a title of a play.
llm = OpenAI(temperature=0.7)
synopsis_template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
synopsis_prompt_template = PromptTemplate(
    input_variables=["title"], template=synopsis_template
)
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)

# This is an LLMChain to write a review of a play given a synopsis.
review_template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
review_prompt_template = PromptTemplate(
    input_variables=["synopsis"], template=review_template
)
review_chain = LLMChain(llm=llm, prompt=review_prompt_template)


chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

if __name__ == "__main__":
    review = chain.run("Tragedy at sunset on the beach")
