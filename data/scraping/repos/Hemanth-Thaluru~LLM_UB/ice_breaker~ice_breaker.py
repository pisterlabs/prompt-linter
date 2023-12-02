from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
# from langchain.llms import HuggingFaceHub
from langchain import HuggingFaceHub

# if __name__=='__main__':
#     print('Hello from Langchain')

information="""
Elon Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate and investor. He is the founder, CEO, and chief engineer of SpaceX; angel investor, CEO and product architect of Tesla, Inc.; owner and CTO of Twitter; founder of the Boring Company; co-founder of Neuralink and OpenAI; and president of the philanthropic Musk Foundation. Musk is the wealthiest person in the world with an estimated net worth, as of July 12, 2023, of around US$239 billion according to the Bloomberg Billionaires Index and $248.8 billion according to Forbes's Real Time B
Musk was born in Pretoria, South Africa, and briefly attended the University of Pretoria before moving to Canada.
"""

summary_template="""
given information {information} about the person I want you to create 
     when he is born?
"""

summary_prompt_template=PromptTemplate(
    input_variables=["information"],
    template=summary_template
)

repo_id='google/flan-t5-xxl'
llm=HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0.5,}
    )
chain = LLMChain(prompt=summary_prompt_template, llm=llm)
print(chain.run(information=information))