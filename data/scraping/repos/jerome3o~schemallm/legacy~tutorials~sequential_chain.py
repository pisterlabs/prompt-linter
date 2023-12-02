from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

from homellm.llm import get_llm

llm = get_llm()


# This is an LLMChain to write a synopsis given a title of a play and the era it is set
# in.
template = """You are a playwright. Given the title of play and the era it is set in, \
it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="synopsis",
)

# This is an LLMChain to write a review of a play given a synopsis.
template = """You are a play critic from the New York Times. Given the synopsis of play\
, it is your job to write a review for that play.

Title: {title}
Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(
    input_variables=["synopsis", "title"], template=template
)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is the overall chain where we run these two chains in sequence.

overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True,
)

output = overall_chain(
    {
        "title": "Tragedy at sunset on the beach",
        "era": "Victorian England",
        "stop": "e",
    }
)

print(output)
print(output)
