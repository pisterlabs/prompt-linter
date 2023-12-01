from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

physics_template = """ 
    You are a very smart physics professor.\
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question, you say "I don't know". \
        
    Here is a question:
    {input}
    """
    
math_template = """
    You are a very good mathematician. \
    You are great at answering math questions.\ 
    You are so good because you are able to break down hand problems into their component parts,\
    answer the component parts, and then put them together to answer the boraider question. \     

    Here is a question:
    {input}
    """   
         
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering question about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering question about math",
        "prompt_template": math_template,
    }
]

def route():
    # 创建 llm
    llm = OpenAI()
    
    # 目的地链
    destination_chains = {}
    for p_info in prompt_infos:
        # 创建 prompt
        prompt = PromptTemplate(
            template=p_info["prompt_template"],
            input_variables=["input"],
        )
        
        # 创建 llm chain
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
        )
        
        # 添加 chain
        destination_chains[p_info["name"]] = chain
        
    default_chain = ConversationChain(
        llm=llm,
        output_key="text",
    )
    
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    
    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )
    
    print(chain.run("What is block body radiation?"))