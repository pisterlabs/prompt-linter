from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

def giveNameForCompany():
    # 初始化 llm
    llm = OpenAI(temperature=0.9)
    
    # 设置 Prompt
    prompt = PromptTemplate(
        input_variables=['product_name'],
        template="What is a good name for a company that males {product_name}?"
    )
    
    # 初始化链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 执行链
    return chain.run("cars")    

def giveNameForProduct():
    # 初始化 llm
    llm = OpenAI(temperature=0.9)
    
    # 设置 Prompt
    prompt = PromptTemplate(
        input_variables=['company_name', 'product_name'],
        template="What is a good name for a {company_name} {product_name}?"
    )
    
    # 初始化链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 执行链
    return chain.run({
        "company_name": "Tesla",
        "product_name": "car"
    })
    
def giveNameForProduct2() -> str:
    # 初始化 llm
    chat_llm = ChatOpenAI(temperature=0.9)
    
    # 设置 Prompt
    human_message_prompt = HumanMessagePromptTemplate(
        prompt= PromptTemplate(
            template="What is a good name for a {company_name} {product_name}?",
            input_variables=['company_name', 'product_name']
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    
    # 初始化链
    chain = LLMChain(llm=chat_llm, prompt=chat_prompt_template)
    
    # 执行链
    return chain.run({
        "company_name": "Tesla",
        "product_name": "car"
    })