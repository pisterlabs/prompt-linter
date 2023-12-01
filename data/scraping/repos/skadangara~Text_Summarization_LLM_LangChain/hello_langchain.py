import os
# import langchain libraries
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain


## Read person details from the text file here...

input_file_name = "person_info.txt"
with open(input_file_name) as f:
    person_info = f.readlines()
f.close()

information = person_info

if __name__ == "__main__":
    print("Person SUmmary suing LLM")
    open_api_key = os.environ["OPENAI_API_KEY"]
    
    # Defining a summary template...
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """
    # Creating a summary prompt template...
    summary_prompt_template = PromptTemplate(input_variables=["information"],template=summary_template)
    
    # Initialising llm
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # Initialising langchain
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    # Running langchain
    print(chain.run(information=information))