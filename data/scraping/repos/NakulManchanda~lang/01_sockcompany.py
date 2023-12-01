import logging
import os
from dotenv import load_dotenv
load_dotenv()

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
SERP_API_KEY=os.environ["SERP_API_KEY"]


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


llm = OpenAI(openai_api_key=OPENAI_API_KEY)
# lang chain app
llm = OpenAI(temperature=0.9)
# text = "What would be a good company name for a company that makes colorful socks?"
# print(llm(text))

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
print(prompt.format(product="colorful socks"))
chain = LLMChain(llm=llm, prompt=prompt)
chain.run("colorful socks")
