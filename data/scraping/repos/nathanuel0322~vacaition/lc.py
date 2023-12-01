from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

chat = ChatAnthropic()
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print(chain.run({"product": "colorful socks"}))

# Second chain
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)
chain_two = LLMChain(llm=chat, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)