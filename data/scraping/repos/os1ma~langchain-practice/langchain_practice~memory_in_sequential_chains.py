from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.memory import SimpleMemory
from langchain.prompts import PromptTemplate
from util import initialize

initialize()

# This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
llm = OpenAI(temperature=0.7)
template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=0.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

llm = OpenAI(temperature=0.7)
template = """You are a social media manager for a theater company.  Given the title of play, the era it is set in, the date,time and location, the synopsis of the play, and the review of the play, it is your job to write a social media post for that play.

Here is some context about the time and location of the play:
Date and Time: {time}
Location: {location}

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:
{review}

Social Media Post:
"""
prompt_template = PromptTemplate(
    input_variables=["synopsis", "review", "time", "location"], template=template
)
social_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="social_post_text")

overall_chain = SequentialChain(
    memory=SimpleMemory(
        memories={"time": "December 25th, 8pm PST", "location": "Theater in the Park"}
    ),
    chains=[synopsis_chain, review_chain, social_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["social_post_text"],
    verbose=True,
)

result = overall_chain(
    {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
)
print(result)
