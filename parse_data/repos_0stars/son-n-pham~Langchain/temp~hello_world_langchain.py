# Important lesson learnt:
# Setup environment variables to store API key securely

# The way to setup environment variables is different for different OS

# For Windows, use the following command in the command prompt:
# set OPENAI_API_KEY=sk-xxx

# For Linux, use the following command in the terminal:
# export OPENAI_API_KEY=sk-xxx

# For Mac, use the following command in the terminal:
# export OPENAI_API_KEY=sk-xxx


# For PowerShell, use the following command in the terminal:
# $env:OPENAI_API_KEY="sk-xxx"

# where sk-xxx is your OpenAI API key


from langchain import PromptTemplate  # Importing the PromptTemplate class from the langchain module
from langchain.chat_models import ChatOpenAI  # Importing the ChatOpenAI class from the langchain.chat_models module
from langchain.chains import LLMChain  # Importing the LLMChain class from the langchain.chains module

import os

api_key = os.environ.get('OPENAI_API_KEY')

information = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate and investor. He is the founder, CEO and chief engineer of SpaceX; angel investor, CEO and product architect of Tesla, Inc.; owner and CEO of Twitter, Inc.; founder of the Boring Company; co-founder of Neuralink and OpenAI; and president of the philanthropic Musk Foundation. Musk is the wealthiest person in the world according to the Bloomberg Billionaires Index, and second-wealthiest according to the Forbes's Real Time Billionaires list as of June 2023, primarily from his ownership stakes in Tesla and SpaceX, with an estimated net worth of around $195 billion according to Bloomberg and $207.6 billion according to Forbes.[4][5][6][7]
"""


if __name__ == '__main__':
    print('Hello, World!')  # Printing a message to the console

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template)
    # Creating an instance of the PromptTemplate class with input_variables='information' and template=summary_template

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=api_key)

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))