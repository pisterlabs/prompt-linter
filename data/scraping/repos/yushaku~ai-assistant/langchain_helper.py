from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

def generate_pet_name(animal_type, color):
    llm = OpenAI(temperature=0.7)
    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="suggest me five cool name for my {animal_type} has {pet_color} bread",
    )
    name_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_name,
        output_key="pet_name"
    )
    return name_chain({"animal_type": animal_type, "pet_color": color})

def langchain_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    result = agent.run("what is the average age of a dog?")
    print (result)

if __name__ == "__main__":
    langchain_agent()
    # print(generate_pet_name("cat", "blue"))

