import os
from langchain import PromptTemplate 
from langchain.chat_models import ChatOpenAI 
from langchain.chains import LLMChain

information = """
Elon Musk, (born June 28, 1971, Pretoria, 
South Africa), South African-born American 
entrepreneur who cofounded the electronic-payment 
firm PayPal and formed SpaceX, maker of launch vehicles and spacecraft. He was also one of the first significant investors in, 
as well as chief executive officer of, the electric car manufacturer Tesla. In addition
"""

if __name__ == '__main__':
    print(os.environ['OPENAI_API_KEY'])
    print("Hello langChain!")
    
    # Create the summary template
    summary_template = """
        given the {information} about a person, I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    # Instantiate the promptemplate which takes any argurment for the template and the template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template)
    
    # Instantiate the chat model, with temperature determining how creative the model can be and model name 
    openapi = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # instantiate the chain(the linker which links the chat model with our prompt)
    chain = LLMChain(llm=openapi, prompt=summary_prompt_template)
    
    print(chain.run(information=information))
    