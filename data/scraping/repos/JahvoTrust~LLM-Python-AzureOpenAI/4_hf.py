import os
import openai
from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="Translate English to SQL: {question}"    
# )

# hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
# print(hub_chain.run("What is the average age of the respondents using a mobile device?"))


# second example below:

# Create a completion - qna를 위하여 davinci 모델생성
llm = AzureOpenAI(deployment_name="text-davinci-003")

# hub_llm = HuggingFaceHub(
#     repo_id='gpt2',
#     model_kwargs={'temperature': 0.7, 'max_length': 100}
# )

prompt = PromptTemplate(
    input_variables=["profession"],
    template="You had one job 😡! You're the {profession} and you didn't have to be sarcastic"
)

hub_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
print(hub_chain.run("customer"))
# print(hub_chain.run("politician"))
# print(hub_chain.run("Fintech CEO"))
# print(hub_chain.run("insurance agent"))