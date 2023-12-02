from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# This is an LLMChain to write a synopsis a title of a play. 
llm = OpenAI(temperature=0.8)
template = """
You are a playwright, Given the title of play, it is your job to write a synopsis for that play.

Title: 
{title}
Playwright: This is a synopsis fot the above play:
"""

prompt_template = PromptTemplate(
    input_variables=["title"],
    template=template,
)
synopsis_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)


# This is an LLMChain to write a review of play given a synopsis.
llm = OpenAI(temperature=0.8)
template = """
You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis: 
{synopsis}
Review from a New York Times play critic of the above play:
"""

prompt_template = PromptTemplate(
    input_variables=["synopsis"],
    template=template
)
review_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)


from langchain.chains import SimpleSequentialChain
def review() -> str:
    overall_chain = SimpleSequentialChain(
        chains=[synopsis_chain, review_chain],
        verbose=True,
    )
    
    return overall_chain.run("Tragedy at sunset on the beach")
    