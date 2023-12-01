import langchain
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
load_dotenv()

def find_details(nameofbird):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.7)

    prompt = PromptTemplate(
        input_variables=["bird_name"],
        template="""I want you to act as an Ornithologist. You have access to details of all the birds in the world.
        I will give you a name of a bird and you want to give me all the details you know about that bird with Scientific Name, Family, Plumage, Description, Habitat, Conservation Status.
        Print the result in markdown format with bullet points only with each attribute in a new line.
        BIRD NAME: {bird_name}""",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # print(prompt.format(bird_name=nameofbird))
    details = chain.run(nameofbird)
    # print(details)
    return details