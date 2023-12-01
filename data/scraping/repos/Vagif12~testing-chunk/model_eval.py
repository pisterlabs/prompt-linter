from langchain import LLMChain, OpenAI, Cohere, HuggingFaceHub, PromptTemplate
from langchain.model_laboratory import ModelLaboratory

llms = [
    OpenAI(temperature=0),
    Cohere(model="command-xlarge-20221108", max_tokens=20, temperature=0),
    HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1}),
]

model_lab = ModelLaboratory.from_llms(llms)

print(model_lab.compare("What color is a flamingo?"))

prompt = PromptTemplate(
    template="What is the capital of {state}?", input_variables=["state"]
)
model_lab_with_prompt = ModelLaboratory.from_llms(llms, prompt=prompt)

print(model_lab_with_prompt.compare("New York"))

from langchain import SelfAskWithSearchChain, SerpAPIWrapper

open_ai_llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
self_ask_with_search_openai = SelfAskWithSearchChain(
    llm=open_ai_llm, search_chain=search, verbose=True
)

cohere_llm = Cohere(temperature=0, model="command-xlarge-20221108")
search = SerpAPIWrapper()
self_ask_with_search_cohere = SelfAskWithSearchChain(
    llm=cohere_llm, search_chain=search, verbose=True
)

chains = [self_ask_with_search_openai, self_ask_with_search_cohere]
names = [str(open_ai_llm), str(cohere_llm)]

model_lab = ModelLaboratory(chains, names=names)

print(model_lab.compare("What is the hometown of the reigning men's U.S. Open champion?"))