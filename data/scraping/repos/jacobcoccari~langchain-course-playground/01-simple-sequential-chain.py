from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# import langchain

# langchain.debug = True

# This is an LLMChain to write a synopsis given a title of a play.

synopsis_template = """You are a playwright. Given the title of play, it is your job to write \ 
                        a concise synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""

review_template = """You are a play critic from the New York Times. Given the synopsis of play, \
                    it is your job to write a concise review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""

model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

synopsis_prompt_template = ChatPromptTemplate.from_template(synopsis_template)
synopsis_chain = LLMChain(
    llm=model,
    prompt=synopsis_prompt_template,
)

review_template = ChatPromptTemplate.from_template(review_template)
review_chain = LLMChain(
    llm=model,
    prompt=review_template,
)

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain],
)

# review = overall_chain("strange roomate")
# print(review)

review = overall_chain.apply(
    [
        {"input": "strange roomate"},
        {"input": "pull my finger"},
    ]
)
print(review)
