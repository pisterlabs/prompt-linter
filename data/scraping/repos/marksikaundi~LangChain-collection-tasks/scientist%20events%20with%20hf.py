from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain import PromptTemplate, HuggingFaceHub
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initializing LLM & repo (repo_id and hf with model config parameters)
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(
    repo_id=repo_id,model_kwargs={"temperature": 0.8, "max_new_tokens": 510}
)

# Chain st: Give me more information about a great scientist
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about scientist {name}"
)
chain1 = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    output_key='person'
)

# Chain nd: The date he was born
second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born"
)
chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    output_key='Isaac newton'
)

# Chain td: 10 major events on that day during his existance
third_input_prompt = PromptTemplate(
    input_variables = ['Isaac newton'],
    template = "Mention 10 major events happened around {Isaac newton} in the world"
)
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    output_key='description'
)

#combining chains
celebrity_chain = SequentialChain(
    chains=[chain1,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','Isaac newton','description']
)
data = celebrity_chain({'name':"Isaac Newton"})
print("Name:", data['name'])
print("Date of Birth:", data['Isaac newton'])
print("Description:")
print(data['person'])
print("Historical Events:")
print(data['description'])