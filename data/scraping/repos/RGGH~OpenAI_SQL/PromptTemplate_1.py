from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub

from qdrant_client import QdrantClient
qdrant = QdrantClient(":memory:")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# model selection
llm = OpenAI(

    temperature=0.9,
    model_name="text-davinci-003"
)

# flan_t5 = HuggingFaceHub(
#     repo_id="google/flan-t5-xl",
#     model_kwargs={"temperature":1e-10}
# )

# ----------------------------------------------------

template = """ Question : What is a good name for {company} that makes {product}?

## context : You are based in England and make everything to order

"""

prompt = PromptTemplate(
    input_variables=["company", "product"],
    template=template,
)


chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run({
    'company': "ShiteCo",
    'product': "Beds"
    }))

# DreamCraft Beds Ltd.
